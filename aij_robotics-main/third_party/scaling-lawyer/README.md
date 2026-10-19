# scaling-lawyer (hackathon subset)

Vendored subset for the AIJ robotics hackathon pipeline.

Kept modules:

- `scaling_lawyer.vla` — SmolVLA action-expert training (LeRobot)
- `scaling_lawyer.common` / `subsets` — helpers used by `vla`
- `scaling_lawyer.libero_eval.SmolVLALiberoModel` — LIBERO layout (image/image2) for submission validation

Full upstream lives outside this repo. Re-vendor with `./scripts/vendor_deps.sh` (applies prune automatically).
