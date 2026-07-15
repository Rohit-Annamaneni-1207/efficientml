from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from ui.callbacks import (
        add_technique,
        clear_pipeline,
        move_technique_down,
        move_technique_up,
        remove_technique,
        run_pipeline,
        technique_visibility,
    )
    from ui.components import APP_CSS, APP_THEME, create_layout
except ModuleNotFoundError as exc:
    missing = exc.name or "dependency"
    raise SystemExit(
        f"Missing required dependency: {missing}. "
        "Install Gradio before launching the UI."
    ) from exc


def build_app():
    demo, components = create_layout()

    with demo:
        components["technique_picker"].change(
            fn=technique_visibility,
            inputs=[components["technique_picker"]],
            outputs=[
                components["technique_description"],
                components["lora_group"],
                components["distillation_group"],
                components["qat_group"],
                components["ptq_group"],
            ],
        )

        components["add_button"].click(
            fn=add_technique,
            inputs=[
                components["technique_picker"],
                components["pipeline_state"],
            ],
            outputs=[
                components["pipeline_state"],
                components["pipeline_preview"],
                components["selected_step"],
            ],
        )

        components["move_up_button"].click(
            fn=move_technique_up,
            inputs=[
                components["selected_step"],
                components["pipeline_state"],
            ],
            outputs=[
                components["pipeline_state"],
                components["pipeline_preview"],
                components["selected_step"],
            ],
        )

        components["move_down_button"].click(
            fn=move_technique_down,
            inputs=[
                components["selected_step"],
                components["pipeline_state"],
            ],
            outputs=[
                components["pipeline_state"],
                components["pipeline_preview"],
                components["selected_step"],
            ],
        )

        components["remove_button"].click(
            fn=remove_technique,
            inputs=[
                components["selected_step"],
                components["pipeline_state"],
            ],
            outputs=[
                components["pipeline_state"],
                components["pipeline_preview"],
                components["selected_step"],
            ],
        )

        components["clear_button"].click(
            fn=clear_pipeline,
            outputs=[
                components["pipeline_state"],
                components["pipeline_preview"],
                components["selected_step"],
            ],
        )

        components["run_button"].click(
            fn=run_pipeline,
            inputs=[
                components["base_model"],
                components["dataset"],
                components["epochs"],
                components["learning_rate"],
                components["batch_size"],
                components["device"],
                components["pipeline_state"],
                components["lora_rank"],
                components["lora_alpha"],
                components["lora_targets"],
                components["student_model"],
                components["distill_alpha"],
                components["distill_temperature"],
                components["qat_bits"],
                components["ptq_backend"],
            ],
            outputs=[
                components["progress_panel"],
                components["execution_summary"],
                components["metrics_dashboard"],
                components["comparison_table"],
                components["logs_box"],
                components["benchmark_report"],
                components["model_file"],
            ],
        )

    return demo


if __name__ == "__main__":
    build_app().queue().launch(
        theme=APP_THEME,
        css=APP_CSS,
    )
