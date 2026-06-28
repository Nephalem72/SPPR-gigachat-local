from __future__ import annotations

import uvicorn

from sppr_colab.api import app
from sppr_colab.config import settings


if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)
