from ltx_service.app import create_app
from ltx_service.config import ServiceConfig, parse_service_config

__all__ = ["ServiceConfig", "create_app", "main", "parse_service_config"]


def main() -> None:
    import uvicorn

    config = parse_service_config()
    uvicorn.run(create_app(config), host=config.host, port=config.port)
