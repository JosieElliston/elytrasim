# elytrasim

Minecraft elytra flight physics simulator written in Rust

## Which `runs/` directory is which

Profiles are self-describing -- the header states the objective, the jitter and the roughness
price a schedule was optimized under. `sweep verify` checks a file replays to the rollout its header
records; `sweep verify --optimality` also re-checks the optimality claim, from the file alone.
But the directories are *generations*, and seeding new work from the wrong one silently inherits
a superseded regularizer. What each one was optimized under:

| directory | jitter | rough | notes |
|---|---|---|---|
| `runs/atlas`, `runs/atlas/nsweep` | 0 | `mu` 1e-4, `limit` 85 | **current.** `mth_lut` trig, `algebraic` flight |
| `runs/antichatter` | 0 | `mu` 1e-3, `limit` 85 | current; the stronger price |
| `runs/corpus` | `sigma` 0.1, 8 draws | *(none)* | superseded. Chatter was held off by jitter and a pass budget, before the curvature price existed |
| `runs/veljit` | `sigma` 0.1, 8 draws | *(none)* | superseded; the study that settled the jitter sigma |
| `runs/jitter`, `runs/jitter2` | per-tick *pitch* sigma | *(none)* | a measured dead end, kept for the record -- see the table in `README-sweep.md` |

Accurate as of commit bf5b9e3. The check that does not rot: read the `# jitter` and `# rough`
lines of the profile you are about to seed from. A file with no `# rough` line predates the
curvature price, whatever directory it is in.
