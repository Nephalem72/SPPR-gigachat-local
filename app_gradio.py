from __future__ import annotations

from sppr_colab.config import settings
from sppr_colab.ui import CSS, create_demo


if __name__ == "__main__":
    auth = None
    if settings.ui_username and settings.ui_password:
        auth = (settings.ui_username, settings.ui_password)
    demo = create_demo().queue(default_concurrency_limit=1)
    launch_result = demo.launch(
        server_name=settings.ui_host,
        server_port=settings.ui_port,
        share=settings.ui_share,
        auth=auth,
        show_error=True,
        debug=False,
        prevent_thread_lock=True,
        footer_links=[],
        css=CSS,
    )

    local_url = getattr(launch_result, "local_url", None)
    share_url = getattr(launch_result, "share_url", None)
    if isinstance(launch_result, (tuple, list)):
        if len(launch_result) > 1:
            local_url = launch_result[1]
        if len(launch_result) > 2:
            share_url = launch_result[2]

    print(f"SPPR UI local URL: {local_url}", flush=True)
    print(f"SPPR UI public URL: {share_url}", flush=True)
    if settings.ui_share and not share_url:
        print("SPPR UI public URL was not created. Check Gradio share output above.", flush=True)

    try:
        import time

        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        demo.close()
