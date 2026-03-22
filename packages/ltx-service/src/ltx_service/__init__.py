import os
import signal
import threading
import time
from types import FrameType

from ltx_service.app import create_app
from ltx_service.config import ServiceConfig, parse_service_config

__all__ = ["ServiceConfig", "create_app", "main", "parse_service_config"]

CTRL_C_FORCE_EXIT_TIMEOUT_SECONDS = 1.0


def _force_exit_after_timeout(timeout_seconds: float) -> None:
    time.sleep(timeout_seconds)
    os._exit(130)


def _start_force_exit_timer(timeout_seconds: float) -> threading.Thread:
    thread = threading.Thread(target=_force_exit_after_timeout, args=(timeout_seconds,), daemon=True)
    thread.start()
    return thread


class _ImmediateCtrlCServer:
    def __init__(self, config) -> None:
        import uvicorn

        server_base = uvicorn.Server

        class ImmediateCtrlCServer(server_base):
            def __init__(self, server_config) -> None:
                super().__init__(server_config)
                self._force_exit_timer_started = False

            def handle_exit(self, sig: int, frame: FrameType | None) -> None:
                super().handle_exit(sig, frame)
                if sig != signal.SIGINT:
                    return
                if self.force_exit:
                    os._exit(130)
                if not self._force_exit_timer_started:
                    self._force_exit_timer_started = True
                    _start_force_exit_timer(CTRL_C_FORCE_EXIT_TIMEOUT_SECONDS)

        self._server = ImmediateCtrlCServer(config)

    def run(self) -> None:
        self._server.run()

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        self._server.handle_exit(sig, frame)


def main() -> None:
    import uvicorn

    config = parse_service_config()
    server_config = uvicorn.Config(create_app(config), host=config.host, port=config.port)
    server = _ImmediateCtrlCServer(server_config)
    server.run()
