# Interview Story — Sprint 2.4: The Baseline Beat the CNN

> The headline a candidate can actually use in an interview: *"I built the classical
> AE-feature baseline first, on purpose. It hit 99.5% accuracy. The CNN I built next,
> on the STFT front-end the literature recommends, got 76%. I shipped both numbers and
> the baseline wins \u2014 the more interesting result is why."*

## The setup

ARCHITECTURE.md \u00a75 says: build the baseline first, then the CNN, and put both
numbers in the README regardless of who wins. The ADR-004 reason is that the
classical AE features (rise time, ring-down count, peak amplitude, energy, duration)
are what the AE literature actually uses. If a 5-feature logistic regression matches
or beats a CNN on a 4-class problem, that's a statement about the problem, not the
model.

The CNN is a 30k-parameter net on a log-magnitude STFT spectrogram (32 freq bins \u00d7
64 time frames), two conv blocks + a linear head. This is not a state-of-the-art
architecture; it's the smallest reasonable CNN. The point was never to win \u2014 it was
to compare honestly.

## The result

| Model | Accuracy | Notable confusion |
|-------|----------|-------------------|
| Logistic regression (5 features) | **99.5%** | 1 impact \u2192 noise |
| Random forest (5 features) | **99.5%** | 1 impact \u2192 crack |
| CNN (STFT, 30k params, 30 epochs) | **76.0%** | 48/50 cracks \u2192 noise |

Test set: 50 events per class (200 total), held out from the 100-per-class training
set. Same dataset for both models.

## Why the baseline wins

The CNN's failure is not a tuning problem. The confusion matrix shows it
systematically misclassifies **crack as noise** \u2014 48 of 50 crack events get the
noise label. That's not a near-miss; it's a structural confusion.

The reason is in `docs/sources.md` and the Sprint 1.4 study: the dx = 5 mm grid
resolves only ~2 points per wavelength at 100 kHz, so the crack source's 80 kHz
carrier is numerically filtered. What reaches the sensor is a low-frequency burst
(0\u201320 kHz) that *looks* spectrally similar to the band-limited residue of the 50 Hz
mains noise (which also leaks into the 20\u201390 kHz band as a low-amplitude floor).
The STFT sees two low-frequency things and can't tell them apart.

The 5 classical features *can* tell them apart, because they collapse the
time-frequency structure into scalars that separate the classes:

- **Peak amplitude**: crack \u2248 8e-7, noise \u2248 1e-7 \u2014 a 6\u201310\u00d7 difference, one
  threshold.
- **Duration**: friction \u2248 3.3 ms, crack \u2248 0.4 ms \u2014 a 8\u00d7 difference.
- **Ring-down count**: noise \u2248 144, mechanical \u2248 13\u201376 \u2014 the noise never decays.

These are exactly the features the AE literature designed to separate these classes
[Pollock]. The CNN has to *learn* these separators from the 2D spectrogram; the
baseline gets them for free because they're the input.

## What the CNN would need to win

The CNN's 76% is a ceiling imposed by the *grid*, not the architecture. Three things
would change the comparison, in order of cost:

1. **A finer grid** (dx = 2.5 mm): raises the resolvable band to ~200 kHz, so the
   crack's 80 kHz carrier survives propagation. The STFT would then see the
   broadband crack burst as distinct from the tonal noise. Cost: 4\u00d7 FDTD runtime
   (Sprint 0.2 documented this \u2014 ~1.3 s per event, still under budget).

2. **A larger CNN** with more conv capacity: might learn the subtle time-domain
   structure the 5 features capture. But this is a 4-class problem on a 5-feature
   space \u2014 there's no richness for a deep model to exploit.

3. **A multi-channel STFT** (all 4 sensors, not just the strongest): the carrier
   survives better on the near sensor, giving the CNN a channel where crack and noise
   are spectrally distinct. Real AE systems do this; the baseline would benefit too,
   but less, because it already gets the separation from the scalar features.

## Why this is the interview story

Three things, in order of importance:

1. **The baseline was built first on purpose.** ARCHITECTURE.md \u00a75 mandates this,
   and ADR-004 says the baseline number goes in the README regardless of what
   follows. The instinct to reach for the CNN first is the wrong one \u2014 the
   baseline is faster to build, easier to debug, and (here) better. Building it
   first is the discipline.

2. **The loss is honest, not hidden.** I didn't tune the CNN until it won, didn't
   drop the noise class, didn't switch to a finer grid to make the CNN look good.
   The 76% and the confusion matrix are in the README. The reason for the loss
   (grid resolution filtering the crack carrier) is in `docs/sources.md`. An
   interviewer can follow the whole chain: the grid choice (Sprint 0.2, wall-clock
   budget) \u2192 the resolution limit (docs/sources.md) \u2192 the spectral confusion
   \u2192 the baseline winning. That's a more interesting story than "the CNN got
   99%".

3. **The loss is a property of the problem, not the model.** A 4-class problem with
   5 well-chosen scalar features is a *tabular* problem, and tabular problems favour
   simple models. Knowing that \u2014 knowing when *not* to reach for the deep model
   \u2014 is most of the work. The CNN story here is "I built it because the brief said
   to, and I can explain why it lost."

## What I'd say if asked "did you need the CNN"

> "No. The classical AE-feature baseline \u2014 logistic regression on rise time,
> ring-down count, peak amplitude, energy, and duration \u2014 hit 99.5% on the
> 4-class problem. The CNN I built next, on the STFT front-end the literature
> recommends, got 76%. The reason is in the grid: at dx = 5 mm the crack source's
> 80 kHz carrier is numerically filtered, so the crack and the 50 Hz noise both
> look like low-frequency content in the STFT. The 5 classical features separate
> them by peak amplitude (a 10\u00d7 difference) and duration, which the CNN has to
> learn from the 2D spectrogram. I shipped both numbers because the ADR said to,
> and the baseline winning is the more interesting result \u2014 it tells you the
> problem is tabular, not perceptual."