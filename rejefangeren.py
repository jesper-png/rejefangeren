#!/usr/bin/env python3
"""Rejefangeren: holder øje med din siddestilling via webcam og skælder dig ud, når du sidder som en reje.

Tilstande:
  python rejefangeren.py              # kører i baggrunden med menulinje-ikon og rød boks
  python rejefangeren.py --preview    # testvindue med kamerabillede, punkter og målinger
  python rejefangeren.py --kameraer   # viser tilsluttede kameraer
  python rejefangeren.py --selvtest   # læser 30 billeder og udskriver målingerne
"""
import argparse
import json
import math
import random
import signal
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

HER = Path(__file__).resolve().parent
MODEL = HER / "models" / "pose_landmarker_lite.task"
DATA = Path.home() / ".rejefangeren"
GAMMEL_DATA = Path.home() / ".holdningsvagt"  # navnet før omdøbningen
CONFIG = DATA / "config.json"
EVENTS = DATA / "events.jsonl"

# Hvor langt fra den kalibrerede gode stilling før det tæller som dårligt
DEFAULT_THRESHOLDS = {
    "naer": 0.15,      # ansigtet er 15 % større end kalibreret = hovedet tættere på skærmen
    "sunket": 0.18,    # lodret afstand næse til skuldre er faldet 18 % (målt i skulderbredder)
    "hoved_ned": 0.6,  # næsen er sunket 0,6 øjenafstande nedad i billedet
    "skaev": 8.0,      # skulderlinjen hælder 8 grader mere end kalibreret
    "side": 1.5,       # hovedet er flyttet 1,5 øjenafstande (ca. 10 cm) til siden
    "skuldre": 0.05,   # skuldrene ser 5 % smallere ud i forhold til hovedet = rullet frem
}
LABELS = {
    "naer": "for tæt på skærmen",
    "sunket": "sammensunket",
    "hoved_ned": "hovedet hænger",
    "skaev": "skæv",
    "side": "lænet ud til siden",
    "skuldre": "skuldrene er rullet frem",
}

# Skældud. Der vælges en tilfældig besked for den værste fejl, hver gang boksen popper op.
SKAELD_TITEL = ["Rejealarm!", "Fanget, reje!", "Rejefangeren har dig", "Op med dig, reje", "Krumrygget reje!"]
SKAELD = {
    "naer": [
        "Næsen er næsten inde i skærmen. Læn dig tilbage.",
        "Du er ikke en reje i et akvarium. Væk fra glasset.",
        "Lidt tættere, og du kan slikke skærmen. Tilbage med dig.",
        "Skærmen bider ikke, men din nakke brokker sig snart. Tilbage.",
    ],
    "sunket": [
        "Du sidder som en pillet reje. Op med dig.",
        "Næste gang serverer jeg dig på rugbrød med mayonnaise.",
        "Rygraden er ikke pynt. Brug den.",
        "Er du ved at blive til en hummer? Ret ryggen.",
        "Krum, lyserød og snart kogt. Ret dig op.",
    ],
    "hoved_ned": [
        "Hovedet hænger som på en kogt reje. Løft det.",
        "Der er intet spændende nede i skødet. Hagen op.",
        "Du ligner en reje, der er faldet i søvn. Hovedet op.",
    ],
    "skaev": [
        "Du hælder som en reje på et skråt fad. Ret skuldrene.",
        "Skæv reje fanget. Lige skuldre, tak.",
        "Du sidder som en bøjet rejehale. Ret dig ud.",
    ],
    "side": [
        "Du kravler ud til siden som en krabbe. Tilbage på midten.",
        "Skærmen er bred, men det er din ryg ikke. Tilbage i stolen.",
        "Rejer svømmer ikke sidelæns. Tilbage på midten.",
    ],
    "skuldre": [
        "Skuldrene er krøllet sammen som en rejehale. Træk dem tilbage.",
        "Åbn brystet, reje. Skuldrene tilbage og ned.",
        "Du folder dig sammen som en reje i en pose. Skuldrene tilbage.",
    ],
}
ROS_TITEL = ["Godt. Du er et menneske igen.", "Sluppet fri. Denne gang.", "Flot. Ingen rejer her.",
             "Sådan. Rank som en flagstang."]


def skaeld(r):
    """Tilfældig skældud for den måling, der er længst over grænsen."""
    vaerst = max(r, key=r.get) if r else None
    if vaerst is None or r[vaerst] < 1:
        return random.choice(SKAELD_TITEL), "Ret dig op."
    return random.choice(SKAELD_TITEL), random.choice(SKAELD[vaerst])
KALIBRERINGSTID = 3.0
ALARM_EFTER = 4.0     # sekunder med dårlig stilling før boksen kommer
SLIP_EFTER = 1.5      # sekunder med god stilling før boksen forsvinder
FEEDBACK_SEK = 12.0   # så længe boksen bliver stående efter du har rettet dig op
GLAT_SEK = 1.0        # målingerne udglattes over ca. så mange sekunder
CAMERA_SETTINGS_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera"


# ---------- konfiguration og log ----------

def flyt_gammel_data():
    if GAMMEL_DATA.exists() and not DATA.exists():
        GAMMEL_DATA.rename(DATA)


def load_config():
    try:
        cfg = json.loads(CONFIG.read_text())
    except (OSError, ValueError):
        cfg = {}
    cfg.setdefault("thresholds", dict(DEFAULT_THRESHOLDS))
    for k, v in DEFAULT_THRESHOLDS.items():
        cfg["thresholds"].setdefault(k, v)
    cfg.setdefault("baseline", None)
    if "baselines" not in cfg:
        # fra før der var en kalibrering pr. kamera
        cfg["baselines"] = {cfg.get("kamera") or "ukendt": cfg["baseline"]} if cfg["baseline"] else {}
    return cfg


def save_config(cfg):
    DATA.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))


def log_event(kind, **data):
    DATA.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a") as f:
        f.write(json.dumps({"t": time.time(), "kind": kind, **data}, ensure_ascii=False) + "\n")


# ---------- målinger ----------

def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def maal(lm, w, h):
    """Udregn rå målinger (i pixels/grader) fra pose-punkterne. None hvis ansigtet ikke ses."""
    def p(i):
        return (lm[i].x * w, lm[i].y * h, lm[i].visibility)

    nose, leye, reye = p(0), p(2), p(5)
    lmo, rmo = p(9), p(10)
    lsh, rsh = p(11), p(12)
    if min(nose[2], leye[2], reye[2]) < 0.5:
        return None
    m = {"oejne": _dist(leye, reye), "naese_y": nose[1], "naese_x": nose[0]}
    if min(lmo[2], rmo[2]) > 0.5:
        # øjne-til-mund er lodret og bliver ikke mindre, når du drejer hovedet mod siden af skærmen
        m["ansigt_h"] = _dist(((leye[0] + reye[0]) / 2, (leye[1] + reye[1]) / 2),
                              ((lmo[0] + rmo[0]) / 2, (lmo[1] + rmo[1]) / 2))
    if lsh[2] > 0.5 and rsh[2] > 0.5:
        bredde = _dist(lsh, rsh)
        if bredde > 1:
            midt_y = (lsh[1] + rsh[1]) / 2
            m["hals"] = (midt_y - nose[1]) / bredde
            m["skulder_vinkel"] = math.degrees(math.atan2(lsh[1] - rsh[1], abs(lsh[0] - rsh[0])))
            m["skulder_bredde"] = bredde
    return m


def afvigelser(m, base):
    """Afvigelse fra den gode stilling pr. måling, i samme enhed som tærsklerne."""
    if not m or not base:
        return {}
    # Drejer du hovedet mod siden, ser øjnene tættere sammen ud, og kigger du ned, bliver ansigtet
    # kortere. Den største af de to størrelser er derfor det bedste bud på, hvor tæt du er.
    stoerrelse = m["oejne"] / base["oejne"]
    if "ansigt_h" in m and "ansigt_h" in base:
        stoerrelse = max(stoerrelse, m["ansigt_h"] / base["ansigt_h"])
    d = {
        "naer": stoerrelse - 1,
        "hoved_ned": (m["naese_y"] - base["naese_y"]) / base["oejne"],
    }
    if "naese_x" in base:
        d["side"] = abs(m["naese_x"] - base["naese_x"]) / base["oejne"]
    if "skulder_bredde" in m and "skulder_bredde" in base:
        # Ruller skuldrene frem, ser de smallere ud. Hovedets størrelse fjerner effekten af,
        # at du kommer tættere på eller længere fra kameraet.
        d["skuldre"] = 1 - (m["skulder_bredde"] / base["skulder_bredde"]) / stoerrelse
    if "hals" in m and "hals" in base:
        d["sunket"] = 1 - m["hals"] / base["hals"]
        d["skaev"] = abs(m["skulder_vinkel"] - base["skulder_vinkel"])
    return d


def baseline_til(cfg, kameranavn):
    for navn, base in cfg["baselines"].items():
        if navn == kameranavn or navn.lower() in kameranavn.lower():
            return base
    return None


def skift_kamera(cfg, kameranavn):
    """Brug den kalibrering, der hører til kameraet (hver placering har sin egen vinkel).
    Returnerer True, hvis kameraet skal kalibreres."""
    cfg["baseline"] = base = baseline_til(cfg, kameranavn)
    if not base or "naese_x" not in base:
        return True
    # kalibreret før skulder-målingen fandtes, selvom skuldrene var synlige
    return "hals" in base and "skulder_bredde" not in base


def gem_kalibrering(cfg, kameranavn, base):
    cfg["baselines"][kameranavn] = base
    cfg["baseline"] = base
    save_config(cfg)
    log_event("kalibrering", kamera=kameranavn, baseline=base)


def ratios(dev, thresholds):
    """>= 1 betyder at målingen er over tærsklen."""
    return {k: v / thresholds[k] for k, v in dev.items()}


class Vurdering:
    """Gør enkeltbilleder til en stabil alarm: dårlig i ALARM_EFTER sek tænder, god i SLIP_EFTER slukker."""

    def __init__(self):
        self.alarm = False
        self.daarlig_siden = None
        self.god_siden = None

    def opdater(self, r, nu):
        daarlig = any(v >= 1 for v in r.values())
        if daarlig:
            self.god_siden = None
            if self.daarlig_siden is None:
                self.daarlig_siden = nu
            if nu - self.daarlig_siden >= ALARM_EFTER:
                self.alarm = True
        else:
            self.daarlig_siden = None
            if self.god_siden is None:
                self.god_siden = nu
            if nu - self.god_siden >= SLIP_EFTER:
                self.alarm = False
        return self.alarm

    def nulstil(self):
        self.__init__()


class Kalibrering:
    def __init__(self):
        self.start = time.monotonic()
        self.samples = []

    def tilfoej(self, m):
        if m:
            self.samples.append(m)

    def faerdig(self):
        return time.monotonic() - self.start >= KALIBRERINGSTID

    def baseline(self):
        if len(self.samples) < 5:
            return None
        base = {}
        for k in self.samples[0]:
            vals = [s[k] for s in self.samples if k in s]
            if len(vals) >= 0.6 * len(self.samples):
                base[k] = statistics.median(vals)
        return base


# ---------- kamera + pose ----------

def sikr_kamera_adgang():
    """Bed macOS om kamera-adgang og vent på svaret (OpenCV selv giver op, før man når at svare)."""
    import threading
    import AVFoundation as av

    status = av.AVCaptureDevice.authorizationStatusForMediaType_(av.AVMediaTypeVideo)
    if status == 0:  # ikke spurgt endnu
        print("macOS spørger om kamera-adgang. Tryk 'Tillad' i dialogen...")
        svar = threading.Event()
        resultat = []

        def faerdig(ok):
            resultat.append(ok)
            svar.set()

        av.AVCaptureDevice.requestAccessForMediaType_completionHandler_(av.AVMediaTypeVideo, faerdig)
        svar.wait(120)
        status = 3 if resultat and resultat[0] else 2
    if status != 3:
        import subprocess
        subprocess.run(["open", CAMERA_SETTINGS_URL])
        raise RuntimeError(
            "Ingen kamera-adgang. Jeg har åbnet Kamera-indstillingerne: slå Terminal til, "
            "luk Terminal helt (Cmd+Q) og start igen."
        )


def kamera_liste():
    """Kameraerne i den rækkefølge OpenCV nummererer dem (sorteret efter macOS' unikke id).
    Hvert kamera er (navn, er_webcam), hvor iPhone- og skrivebordskameraer ikke tæller som webcam."""
    import AVFoundation as av

    enheder = list(av.AVCaptureDevice.devicesWithMediaType_(av.AVMediaTypeVideo))
    enheder += list(av.AVCaptureDevice.devicesWithMediaType_(av.AVMediaTypeMuxed))
    enheder.sort(key=lambda d: d.uniqueID())
    liste = []
    for d in enheder:
        navn = d.localizedName()
        telefon = (d.respondsToSelector_("isContinuityCamera") and d.isContinuityCamera()) or "iphone" in navn.lower()
        skrivebord = "skrivebord" in navn.lower() or "desk view" in navn.lower()
        liste.append((navn, not telefon and not skrivebord))
    return liste


def find_kamera(navn):
    """Find OpenCV-nummeret for kameraet hvis navn indeholder `navn`. Er det ikke tilsluttet
    (eller er der intet navn), bruges første eksterne webcam, ellers Mac'ens eget kamera.
    iPhone- og skrivebordskameraer vælges aldrig automatisk."""
    liste = kamera_liste()
    if navn:
        for i, (n, _) in enumerate(liste):
            if navn.lower() in n.lower():
                return i, n
    webcams = [(i, n) for i, (n, ok) in enumerate(liste) if ok]
    if not webcams:
        fundne = ", ".join(n for n, _ in liste) or "ingen"
        raise RuntimeError(f"Fandt intet webcam. Tilsluttede kameraer: {fundne}")
    indbyggede = ("macbook", "facetime", "indbygget", "built-in")
    eksterne = [(i, n) for i, n in webcams if not any(x in n.lower() for x in indbyggede)]
    return (eksterne or webcams)[0]


def find_kameraer():
    print("Tilsluttede kameraer (brug en del af navnet med --kamera):")
    for n, ok in kamera_liste():
        print(f"  {n}" + ("" if ok else "  (vælges ikke automatisk)"))


class Vagt:
    def __init__(self, kamera, kraev_kamera=True):
        sikr_kamera_adgang()
        self.kamera = kamera
        self.navn = None
        self.cap = None
        self.fejl = 0
        self.sidste_aabning = 0.0
        self.pauset = False
        self.sidste_besked = None
        self.aabn_kamera()
        # Baggrundstilstanden venter bare på kameraet (fx ved login, før det er sat i)
        if self.cap is None and kraev_kamera:
            raise RuntimeError(
                f"Kunne ikke åbne kameraet. Giv Terminal adgang til kameraet: {CAMERA_SETTINGS_URL}"
            )
        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(opts)
        self.t0 = time.monotonic()
        self.sidste_ts = -1
        self.glat = None
        self.glat_tid = None

    def aabn_kamera(self):
        """Åbn kameraet. Lykkes det ikke (fx stikket er trukket ud), prøver laes() igen senere."""
        self.slip_kamera()
        self.pauset = False
        self.fejl = 0
        self.sidste_aabning = time.monotonic()
        try:
            nr, navn = find_kamera(self.kamera)
        except RuntimeError as e:
            self.besked(f"{e} Prøver igen om lidt.")
            return
        cap = cv2.VideoCapture(nr)
        if not cap.isOpened():
            cap.release()
            self.besked(f"Kunne ikke åbne {navn}. Prøver igen om lidt.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap = cap
        self.navn = navn
        self.besked(f"Bruger kamera: {navn}")

    def besked(self, tekst):
        # skriv kun i loggen, når noget ændrer sig, så den ikke fyldes op mens kameraet er væk
        if tekst != self.sidste_besked:
            print(tekst)
            self.sidste_besked = tekst

    def slip_kamera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.navn = None

    def pause_kamera(self):
        """Giv kameraet fri, mens testvinduet bruger det, uden at prøve at tage det tilbage."""
        self.slip_kamera()
        self.pauset = True

    def laes(self):
        """Returnerer (billede, udglattede målinger eller None, pose-punkter eller None)."""
        if self.cap is None:
            if not self.pauset and time.monotonic() - self.sidste_aabning > 5:
                self.aabn_kamera()
            return None, None, None
        ok, frame = self.cap.read()
        if not ok:
            # Kameraet går i stå, hvis stikket trækkes ud, eller et andet program har lånt det.
            self.fejl += 1
            if self.fejl >= 3 and time.monotonic() - self.sidste_aabning > 5:
                self.besked("Kameraet svarer ikke, åbner det igen")
                self.aabn_kamera()
            return None, None, None
        self.fejl = 0
        h, w = frame.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ts = max(int((time.monotonic() - self.t0) * 1000), self.sidste_ts + 1)
        self.sidste_ts = ts
        res = self.landmarker.detect_for_video(img, ts)
        if not res.pose_landmarks:
            self.glat = None
            return frame, None, None
        lm = res.pose_landmarks[0]
        m = maal(lm, w, h)
        if m is None:
            self.glat = None
        elif self.glat is None or self.glat.keys() != m.keys():
            self.glat = m
        else:
            # tidsbaseret udglatning (~1 sek), så tallene ikke hopper uanset billedhastighed
            dt = time.monotonic() - self.glat_tid
            a = 1 - math.exp(-dt / GLAT_SEK)
            self.glat = {k: (1 - a) * self.glat[k] + a * m[k] for k in m}
        self.glat_tid = time.monotonic()
        return frame, self.glat, lm

    def luk(self):
        self.slip_kamera()
        self.landmarker.close()


# ---------- læring via knapperne ----------

def forkert_alarm(cfg, r):
    """Brugeren sad fint: løft tærsklen for de målinger der udløste, så netop denne stilling er OK."""
    t = cfg["thresholds"]
    for k, v in r.items():
        if v >= 1:
            t[k] = min(DEFAULT_THRESHOLDS[k] * 3, t[k] * v * 1.1)
    save_config(cfg)
    log_event("forkert_alarm", ratios=r, thresholds=t)


def korrekt_alarm(cfg, r):
    """Alarmen var rigtig: gør de udløsende målinger en smule mere følsomme."""
    t = cfg["thresholds"]
    for k, v in r.items():
        if v >= 1:
            t[k] = max(DEFAULT_THRESHOLDS[k] * 0.6, t[k] * 0.97)
    save_config(cfg)
    log_event("korrekt_alarm", ratios=r, thresholds=t)


def grunde(r):
    return [LABELS[k] for k, v in sorted(r.items(), key=lambda kv: -kv[1]) if v >= 1]


def grund_linje(r):
    g = grunde(r)
    return "Fanget: " + ", ".join(g) if g else ""


# ---------- kommandolinje-tilstande ----------



def selvtest(kamera):
    vagt = Vagt(kamera)
    fundet, samples = 0, []
    for _ in range(30):
        _, m, _ = vagt.laes()
        if m:
            fundet += 1
            samples.append(m)
    vagt.luk()
    print(f"Person fundet i {fundet}/30 billeder")
    if samples:
        for k in samples[-1]:
            vals = [s[k] for s in samples if k in s]
            print(f"  {k}: {statistics.median(vals):.2f}")
        if "hals" not in samples[-1]:
            print("  OBS: skuldrene ses ikke, så kun hoved-målinger bruges")


JUSTER_TRIN = 1.1  # hvert klik på + eller - ændrer grænsen 10 %
FONT = cv2.FONT_HERSHEY_SIMPLEX
# Testvinduets skrift kan ikke vise æ/ø/å
KORT_NAVN = {"naer": "Taet paa skaerm", "sunket": "Sammensunket", "hoved_ned": "Hoved haenger",
             "skaev": "Skaev", "side": "Til siden", "skuldre": "Skuldre frem"}


def juster_graense(cfg, k, retning):
    """+ løfter grænsen (tillad mere), - sænker den (strengere)."""
    t = cfg["thresholds"]
    ny = t[k] * JUSTER_TRIN if retning > 0 else t[k] / JUSTER_TRIN
    t[k] = min(DEFAULT_THRESHOLDS[k] * 5, max(DEFAULT_THRESHOLDS[k] * 0.2, ny))
    save_config(cfg)
    log_event("manuel_justering", maaling=k, graense=t[k])


def tegn_knap(frame, x, y, tegn):
    cv2.rectangle(frame, (x, y), (x + 26, y + 22), (90, 90, 90), -1)
    cv2.putText(frame, tegn, (x + 7, y + 17), FONT, 0.6, (255, 255, 255), 2)
    return (x, y, x + 26, y + 22)


def preview(kamera):
    cfg = load_config()
    vagt = Vagt(kamera)
    vurd = Vurdering()
    kalib = None
    aktivt_kamera = None
    knapper = []  # (x1, y1, x2, y2, måling, retning) fra seneste billede
    dev = {}
    frosset = False
    vindue = "Rejefangeren (justering)"
    cv2.namedWindow(vindue)

    def klik(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for x1, y1, x2, y2, k, retning in knapper:
            if x1 <= x <= x2 and y1 <= y <= y2:
                juster_graense(cfg, k, retning)
                print(f"{k}: grænse {cfg['thresholds'][k]:.2f}")

    cv2.setMouseCallback(vindue, klik)
    print("Testvindue: klik - / + ud for en måling for at justere den, "
          "mellemrum for at fryse tallene, k for at kalibrere (sid rigtigt), q for at afslutte.")
    while True:
        frame, m, lm = vagt.laes()
        if frame is None:
            break
        h, w = frame.shape[:2]
        if lm:
            for i in (0, 2, 5, 9, 10, 11, 12):
                cv2.circle(frame, (int(lm[i].x * w), int(lm[i].y * h)), 5, (0, 255, 255), -1)
            cv2.line(frame, (int(lm[11].x * w), int(lm[11].y * h)), (int(lm[12].x * w), int(lm[12].y * h)), (0, 255, 255), 2)
        frame = cv2.flip(frame, 1)  # spejlvendt, så det føles naturligt
        if vagt.navn and vagt.navn != aktivt_kamera:
            aktivt_kamera = vagt.navn
            kalib = Kalibrering() if skift_kamera(cfg, aktivt_kamera) else None
            vurd.nulstil()

        status = None
        if kalib:
            kalib.tilfoej(m)
            dev = {}
            status = ("Sid rigtigt... kalibrerer", (255, 200, 0))
            if kalib.faerdig():
                base = kalib.baseline()
                if base:
                    gem_kalibrering(cfg, aktivt_kamera, base)
                    print("Kalibreret:", {k: round(v, 2) for k, v in base.items()})
                else:
                    print("Kalibrering fejlede: kunne ikke se dig tydeligt nok. Prøv igen med k.")
                kalib = None
                vurd.nulstil()
        elif m is None and not frosset:
            dev = {}
            status = ("Ingen person fundet", (200, 200, 200))
        elif not frosset:
            dev = afvigelser(m, cfg["baseline"])
            if vurd.opdater(ratios(dev, cfg["thresholds"]), time.monotonic()):
                cv2.rectangle(frame, (w - 170, 10), (w - 10, 70), (0, 0, 255), -1)
                cv2.putText(frame, "REJEALARM", (w - 160, 48), FONT, 0.7, (255, 255, 255), 2)

        # Justeringspanel ved siden af kamerabilledet, så det ikke dækker dig.
        # Procenten regnes ud fra den aktuelle grænse, så et klik på - / + ses med det samme.
        r = ratios(dev, cfg["thresholds"])
        pw = 540
        lærred = np.zeros((max(h, 60 + 34 * len(DEFAULT_THRESHOLDS) + 40), w + pw, 3), np.uint8)
        lærred[:] = (40, 40, 40)
        lærred[:h, :w] = frame
        frame = lærred
        ox = w + 10
        cv2.putText(frame, "- strengere   + tillad mere", (ox, 24), FONT, 0.5, (220, 220, 220), 1)
        cv2.putText(frame, "mellemrum: frys tal   k: kalibrer   q: luk", (ox, 46), FONT, 0.45, (180, 180, 180), 1)
        knapper.clear()
        y = 62
        bx, bw = ox + 190, 220  # bjælke: 0 % til 200 % af grænsen, stregen midt på er grænsen
        for k in DEFAULT_THRESHOLDS:
            knapper.append((*tegn_knap(frame, ox, y, "-"), k, -1))
            knapper.append((*tegn_knap(frame, ox + 32, y, "+"), k, 1))
            cv2.putText(frame, KORT_NAVN[k], (ox + 68, y + 17), FONT, 0.5, (255, 255, 255), 1)
            cv2.rectangle(frame, (bx, y + 4), (bx + bw, y + 20), (70, 70, 70), -1)
            if k in r:
                v = max(0.0, r[k])
                farve = (0, 0, 255) if v >= 1 else (0, 190, 0)
                cv2.rectangle(frame, (bx, y + 4), (bx + int(min(v, 2) / 2 * bw), y + 20), farve, -1)
                cv2.putText(frame, f"{round(v * 20) * 5:d}%", (bx + bw + 10, y + 18), FONT, 0.55, farve, 2)
            else:
                cv2.putText(frame, "ses ikke", (bx + bw + 10, y + 18), FONT, 0.5, (160, 160, 160), 1)
            cv2.line(frame, (bx + bw // 2, y), (bx + bw // 2, y + 24), (255, 255, 255), 2)
            y += 34
        if frosset:
            status = ("FROSSET - tryk mellemrum", (0, 255, 255))
        if status:
            cv2.putText(frame, status[0], (ox, y + 18), FONT, 0.6, status[1], 2)

        cv2.imshow(vindue, frame)
        tast = cv2.waitKey(1) & 0xFF
        if tast == ord("q"):
            break
        if tast == ord("k"):
            kalib = Kalibrering()
            frosset = False
        if tast == ord(" "):
            frosset = not frosset
    vagt.luk()
    cv2.destroyAllWindows()


# ---------- baggrundstilstand med menulinje og rød boks (macOS) ----------

def overlay(kamera):
    import objc
    from AppKit import (
        NSApplication, NSBox, NSButton, NSColor, NSEvent, NSFont, NSMenu, NSMenuItem, NSPanel,
        NSScreen, NSSound, NSStatusBar, NSTextField,
    )
    from Foundation import NSMakeRect, NSObject, NSTimer

    W, H = 420, 165

    class Styring(NSObject):
        @objc.python_method
        def setup(self):
            self.cfg = load_config()
            self.vagt = Vagt(kamera, kraev_kamera=False)
            self.vurd = Vurdering()
            self.kalib = None
            self.aktivt_kamera = None
            self.pause_til = 0.0
            self.snooze_til = 0.0
            self.test = False
            self.r = {}
            self.alarm_vist = False
            self.alarm_r = {}
            self.feedback_til = None
            self.justering = None
            self.skaeld_tekst = ("", "")
            self.byg_panel()
            self.byg_menu()
            NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(0.12, self, "tick:", None, True)
            return self

        @objc.python_method
        def byg_panel(self):
            self.panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
                NSMakeRect(0, 0, W, H), 1 << 7, 2, False)  # uden kant, aktiverer ikke appen
            self.panel.setLevel_(25)  # over almindelige vinduer
            self.panel.setCollectionBehavior_((1 << 0) | (1 << 4) | (1 << 8))  # alle skriveborde + fuldskærm
            self.panel.setOpaque_(False)
            self.panel.setBackgroundColor_(NSColor.clearColor())
            self.panel.setHasShadow_(True)
            self.panel.setHidesOnDeactivate_(False)
            self.panel.setFloatingPanel_(True)
            cv = self.panel.contentView()
            # NSBox giver en afrundet, farvet baggrund
            self.baggrund = NSBox.alloc().initWithFrame_(NSMakeRect(0, 0, W, H))
            self.baggrund.setBoxType_(4)  # NSBoxCustom
            self.baggrund.setBorderWidth_(0)
            self.baggrund.setCornerRadius_(14)
            self.baggrund.setTitlePosition_(0)  # ingen titel
            cv.addSubview_(self.baggrund)

            self.titel = NSTextField.labelWithString_("")
            self.titel.setFont_(NSFont.boldSystemFontOfSize_(20))
            self.titel.setTextColor_(NSColor.whiteColor())
            self.titel.setFrame_(NSMakeRect(18, H - 44, W - 36, 28))
            self.tekst = NSTextField.wrappingLabelWithString_("")
            self.tekst.setFont_(NSFont.systemFontOfSize_(14))
            self.tekst.setTextColor_(NSColor.whiteColor())
            self.tekst.setFrame_(NSMakeRect(18, 50, W - 36, H - 98))
            self.knap_ja = NSButton.buttonWithTitle_target_action_("Ja, jeg sad dårligt", self, "korrekt:")
            self.knap_ja.setFrame_(NSMakeRect(14, 12, 160, 32))
            self.knap_nej = NSButton.buttonWithTitle_target_action_("Forkert alarm", self, "forkert:")
            self.knap_nej.setFrame_(NSMakeRect(180, 12, 130, 32))
            self.knap_afvis = NSButton.buttonWithTitle_target_action_("Afvis", self, "afvis:")
            self.knap_afvis.setFrame_(NSMakeRect(316, 12, 90, 32))
            for v in (self.titel, self.tekst, self.knap_ja, self.knap_nej, self.knap_afvis):
                cv.addSubview_(v)


        @objc.python_method
        def vis(self, titel, tekst, farve, knapper):
            if not self.panel.isVisible():
                self.placer_panel()
            self.titel.setStringValue_(titel)
            self.tekst.setStringValue_(tekst)
            self.baggrund.setFillColor_(farve)
            self.knap_ja.setHidden_(not knapper)
            self.knap_nej.setHidden_(not knapper)
            self.knap_afvis.setHidden_(not knapper)
            self.panel.orderFrontRegardless()

        @objc.python_method
        def placer_panel(self):
            """Øverst på midten af den skærm, musen er på, så boksen er der, hvor du kigger."""
            mus = NSEvent.mouseLocation()
            skaerm = NSScreen.mainScreen()
            for s in NSScreen.screens():
                f = s.frame()
                if f.origin.x <= mus.x < f.origin.x + f.size.width and f.origin.y <= mus.y < f.origin.y + f.size.height:
                    skaerm = s
            v = skaerm.visibleFrame()
            self.panel.setFrameOrigin_((v.origin.x + (v.size.width - W) / 2, v.origin.y + v.size.height - H - 20))

        @objc.python_method
        def skjul(self):
            self.panel.orderOut_(None)
            self.alarm_vist = False
            self.feedback_til = None

        @objc.python_method
        def byg_menu(self):
            self.item = NSStatusBar.systemStatusBar().statusItemWithLength_(-1)
            self.item.button().setTitle_("🦐")
            menu = NSMenu.alloc().init()
            for titel, sel in (("Kalibrér (sid rigtigt først)", "kalibrer:"),
                               ("Pause 30 min", "pause:"),
                               ("Genoptag", "genoptag:"),
                               ("Justér grænser...", "juster:"),
                               ("Vis boksen (test)", "testboks:"),
                               ("Afslut", "afslut:")):
                mi = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(titel, sel, "")
                mi.setTarget_(self)
                menu.addItem_(mi)
            self.item.setMenu_(menu)

        def tick_(self, timer):
            nu = time.monotonic()
            self.genindlaes_hvis_aendret()
            if self.test:
                return
            if self.justering:
                if self.justering.poll() is None:
                    self.item.button().setTitle_("🦐 ✎")
                    return
                self.justering = None
                self.cfg = load_config()
                self.aktivt_kamera = None  # finder kalibreringen til kameraet igen
                self.vagt.aabn_kamera()
                self.vurd.nulstil()
            if nu < self.pause_til:
                self.item.button().setTitle_("🦐 ⏸")
                return
            _, m, _ = self.vagt.laes()
            if self.vagt.navn and self.vagt.navn != self.aktivt_kamera:
                self.aktivt_kamera = self.vagt.navn
                self.kalib = Kalibrering() if skift_kamera(self.cfg, self.aktivt_kamera) else None
                self.vurd.nulstil()

            if self.kalib:
                self.kalib.tilfoej(m)
                self.vis("Sid rigtigt og ret op", "Rejefangeren lærer, hvordan du ser ud, når du sidder pænt...", NSColor.systemBlueColor(), False)
                if self.kalib.faerdig():
                    base = self.kalib.baseline()
                    self.kalib = None
                    self.vurd.nulstil()
                    self.skjul()
                    if base:
                        gem_kalibrering(self.cfg, self.aktivt_kamera, base)
                    elif not self.cfg["baseline"] and self.aktivt_kamera:
                        self.kalib = Kalibrering()
                return

            if m is None:
                self.item.button().setTitle_("🦐 –")
                self.vurd.nulstil()
                self.efter_alarm(nu)
                return

            self.r = ratios(afvigelser(m, self.cfg["baseline"]), self.cfg["thresholds"])
            alarm = self.vurd.opdater(self.r, nu) and nu >= self.snooze_til
            self.item.button().setTitle_("🦐 ⚠️" if alarm else "🦐 ✓")
            if alarm:
                if not self.alarm_vist:
                    NSSound.soundNamed_("Pop").play()
                    log_event("alarm", ratios=self.r)
                    self.alarm_r = {}
                    self.skaeld_tekst = skaeld(self.r)
                # husk de højeste værdier under alarmen, så feedback lærer af dem og ikke af den rettede stilling
                for k, v in self.r.items():
                    self.alarm_r[k] = max(v, self.alarm_r.get(k, v))
                titel, tekst = self.skaeld_tekst
                tekst = f"{tekst}\n{grund_linje(self.alarm_r)}".strip()
                self.vis(titel, tekst, NSColor.colorWithCalibratedRed_green_blue_alpha_(0.85, 0.12, 0.12, 0.96), True)
                self.alarm_vist = True
                self.feedback_til = None
            else:
                self.efter_alarm(nu)

        @objc.python_method
        def efter_alarm(self, nu):
            """Når du har rettet dig op, bliver boksen stående lidt, så du kan nå at give feedback."""
            if not self.alarm_vist:
                return
            if self.feedback_til is None:
                self.feedback_til = nu + FEEDBACK_SEK
                tekst = f"{grund_linje(self.alarm_r)}\nVar alarmen rigtig? Lukker selv om {FEEDBACK_SEK:.0f} sek."
                self.vis(random.choice(ROS_TITEL), tekst.strip(),
                         NSColor.colorWithCalibratedRed_green_blue_alpha_(0.15, 0.55, 0.25, 0.96), True)
            elif nu >= self.feedback_til:
                log_event("ingen_feedback", ratios=self.alarm_r)
                self.skjul()

        @objc.python_method
        def luk_efter_klik(self, snooze):
            if not self.test:
                self.snooze_til = time.monotonic() + snooze
            self.test = False
            self.vurd.nulstil()
            self.skjul()

        def korrekt_(self, sender):
            if not self.test:
                korrekt_alarm(self.cfg, self.alarm_r)
            self.luk_efter_klik(15)

        def forkert_(self, sender):
            if not self.test:
                forkert_alarm(self.cfg, self.alarm_r)
            self.luk_efter_klik(60)

        def afvis_(self, sender):
            """Luk boksen uden at lære noget af det."""
            if not self.test:
                log_event("afvist", ratios=self.alarm_r)
            self.luk_efter_klik(60)

        def kalibrer_(self, sender):
            self.pause_til = 0
            self.kalib = Kalibrering()

        def pause_(self, sender):
            self.pause_til = time.monotonic() + 30 * 60
            self.skjul()

        def genoptag_(self, sender):
            self.pause_til = 0

        @objc.python_method
        def genindlaes_hvis_aendret(self):
            """Testvinduet gemmer grænser og kalibrering i samme fil; hent dem ind, når filen ændres."""
            try:
                mtime = CONFIG.stat().st_mtime
            except OSError:
                return
            if mtime != getattr(self, "config_mtime", None):
                if getattr(self, "config_mtime", None) is not None:
                    self.cfg = load_config()
                    if self.aktivt_kamera:
                        skift_kamera(self.cfg, self.aktivt_kamera)
                self.config_mtime = mtime

        def juster_(self, sender):
            import subprocess
            if self.justering and self.justering.poll() is None:
                return
            self.vagt.pause_kamera()  # to programmer på samme kamera får det til at gå i stå
            self.skjul()
            self.justering = subprocess.Popen([sys.executable, str(HER / "rejefangeren.py"), "--preview"])

        def testboks_(self, sender):
            self.test = True
            NSSound.soundNamed_("Pop").play()
            titel, tekst = skaeld({"sunket": 1.5})
            self.vis(titel, tekst + "\nTestvisning. Tryk på en knap for at lukke.",
                     NSColor.colorWithCalibratedRed_green_blue_alpha_(0.85, 0.12, 0.12, 0.96), True)

        def afslut_(self, sender):
            self.vagt.luk()
            NSApplication.sharedApplication().terminate_(None)

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(1)  # intet Dock-ikon, kun menulinjen
    styring = Styring.alloc().init().setup()
    signal.signal(signal.SIGINT, lambda *_: styring.afslut_(None))
    print("Rejefangeren kører. Se 🦐 i menulinjen. Ctrl+C eller menuen 'Afslut' stopper den.")
    app.run()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kamera", help="en del af kameraets navn, fx Logitech (gemmes til næste gang)")
    ap.add_argument("--preview", action="store_true", help="testvindue med kamerabillede og målinger")
    ap.add_argument("--kameraer", action="store_true", help="vis tilsluttede kameraer")
    ap.add_argument("--selvtest", action="store_true", help="læs 30 billeder og udskriv målinger")
    ap.add_argument("--nulstil", action="store_true", help="slet kalibrering og lærte tærskler")
    a = ap.parse_args()
    flyt_gammel_data()

    if a.kameraer:
        return find_kameraer()
    cfg = load_config()
    if a.nulstil:
        CONFIG.unlink(missing_ok=True)
        print("Nulstillet. Næste start kalibrerer forfra.")
        return
    if a.kamera is not None and a.kamera != cfg.get("kamera"):
        cfg["kamera"] = a.kamera
        save_config(cfg)
    kamera = cfg.get("kamera") if isinstance(cfg.get("kamera"), str) else None
    try:
        if a.selvtest:
            selvtest(kamera)
        elif a.preview:
            preview(kamera)
        else:
            overlay(kamera)
    except RuntimeError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
