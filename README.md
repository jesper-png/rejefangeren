# 🦐 Rejefangeren

Sidder du krum som en reje foran skærmen? Rejefangeren holder øje med dig gennem dit webcam og skælder dig ud, når du synker sammen, hænger med hovedet eller kryber helt ind i skærmen.

> *"Du sidder som en pillet reje. Op med dig."*
> *"Næste gang serverer jeg dig på rugbrød med mayonnaise."*

Alt kører lokalt på din Mac. Intet billede bliver gemt eller sendt nogen steder, og det er gratis.

## Sådan virker den

1. Du sidder pænt i 3 sekunder, og Rejefangeren lærer, hvordan du ser ud, når du sidder rigtigt.
2. Sidder du derefter dårligt i mere end 4 sekunder, popper en rød boks op (med en lille pop-lyd) på den skærm, du arbejder på.
3. Når du retter dig op, bliver boksen grøn, og du kan fortælle, om alarmen var rigtig:
   - **Ja, jeg sad dårligt**: Rejefangeren bliver en anelse strengere.
   - **Forkert alarm**: du sad fint, så grænsen for netop den måling løftes.
   - **Afvis**: lukker boksen uden at lære noget af det.

Den holder øje med seks ting:

| Måling | Hvad den fanger |
|---|---|
| Tæt på skærm | hovedet er rykket ind mod skærmen (virker også, når du kigger mod siderne af en bred eller buet skærm) |
| Sammensunket | afstanden mellem hoved og skuldre er blevet mindre |
| Hoved hænger | hovedet er sunket ned |
| Skæv | skuldrene hælder |
| Til siden | du læner dig ud mod siden af skærmen |
| Skuldre frem | skuldrene ruller frem, så de ser smallere ud i forhold til hovedet |

## Krav

- En Mac med Apple-chip (M1 eller nyere)
- Et webcam: det indbyggede eller et eksternt oven på skærmen (et eksternt webcam bliver valgt automatisk)
- Python 3.10 til 3.12. Har du den ikke, så installér [Homebrew](https://brew.sh) og kør `brew install python@3.12`

## Installation

Åbn Terminal og kør:

```bash
git clone https://github.com/jesper-png/rejefangeren.git ~/rejefangeren && zsh ~/rejefangeren/installer.command
```

Første gang tager det et par minutter. Så sker der tre ting:

1. Der kommer en 🦐 i menulinjen øverst på skærmen.
2. macOS spørger, om **Python** må bruge kameraet. Tryk **Tillad**.
3. En blå boks beder dig sidde pænt og kigge på midten af skærmen i 3 sekunder.

Fremover starter Rejefangeren automatisk, når du logger ind.

## Brug

Klik på 🦐 i menulinjen:

- **Kalibrér**: sid pænt, og vælg den for at lære din gode stilling forfra
- **Pause 30 min** og **Genoptag**
- **Justér grænser...**: åbner et vindue med kamerabilledet og en bjælke for hver måling. Den hvide streg er grænsen. Klik **-** for at gøre en måling strengere og **+** for at tillade mere. **Mellemrum** fryser tallene, så du kan justere i ro, og **q** lukker vinduet
- **Vis boksen (test)**: se hvordan skældud-boksen ser ud

Symbolet ved rejen viser status: ✓ god stilling, ⚠️ dårlig stilling, – kan ikke se dig, ⏸ pause, ✎ justering.

Arbejder du flere steder med forskellige webcams, husker Rejefangeren en kalibrering for hvert kamera og skifter selv, når du sætter et andet i. Første gang et nyt kamera dukker op, kalibrerer den automatisk. iPhone-kameraer bliver aldrig valgt af sig selv.

Kalibrering, grænser og en log ligger i `~/.rejefangeren`.

## Opdatering og afinstallation

Opdatér til nyeste version:

```bash
cd ~/rejefangeren && git pull && zsh installer.command
```

Afinstallér:

```bash
zsh ~/rejefangeren/afinstaller.command
```

## Teknik

Rejefangeren bruger Googles [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) til at finde ansigt og skuldre i kamerabilledet, OpenCV til kameraet og PyObjC til menulinjen og pop-up'en. Modellen hentes fra Google under installationen.

## Licens

MIT. Se [LICENSE](LICENSE).
