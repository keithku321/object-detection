# Gradio demo that mounts one tab per task. Run from the repository root:   python -m demo.app
# Each task folder provides demoTab.py with buildTab() -> gradio.Blocks (plans/05-team-workflow.md); this file only arranges them.
# A tab whose code or packages are missing is replaced by a message, so one unfinished task never blocks the demo of the others.
# Steward: plan A owner. Do not make this demo public before the team and supervisor have decided on the AGPL question (plans/decisions.md D9).
import argparse
import importlib

# Tabs in display order: tab title -> module that provides buildTab()
taskTabs = {
    "Underwater": "underwater.demoTab",
    "Adverse weather": "weather.demoTab",
    "Small objects": "smallobject.demoTab",
    "Camouflage": "camoedge.demoTab",
}


# Build the whole application; returns a gradio.Blocks object
def buildApp():
    import gradio as gr
    with gr.Blocks(title="Robust object detection under challenging conditions") as app:
        gr.Markdown("# Robust object detection under challenging conditions\nFour subtasks, one tab each. Pick a model in a tab to compare the baseline with the proposed method.")
        for tabTitle, moduleName in taskTabs.items():
            with gr.Tab(tabTitle):
                try:
                    importlib.import_module(moduleName).buildTab().render()
                except Exception as error:  # a broken tab must not take the other tabs down
                    gr.Markdown(f"This tab could not be loaded: `{type(error).__name__}: {error}`")
    return app


# Command line entry point; --share creates a temporary public link, which you should only use for a private demonstration
def main():
    parser = argparse.ArgumentParser(description="Start the demo app")
    parser.add_argument("--share", action="store_true", help="create a temporary public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    arguments = parser.parse_args()
    buildApp().launch(share=arguments.share, server_port=arguments.port)


if __name__ == "__main__":
    main()
