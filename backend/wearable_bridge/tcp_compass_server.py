import asyncio
import json
from typing import Dict, Optional


class WearableCompassTCPServer:
    # Receives 120ms DoA bearing updates over TCP from the Raspberry Pi Zero 2 W neckband.
    def __init__(self, host: str = "0.0.0.0", port: int = 8766):
        self.host = host
        self.port = port
        self.server: Optional[asyncio.AbstractServer] = None
        self.latest_state: Optional[Dict[str, float]] = None

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                data = json.loads(line.decode("utf-8"))
                if data.get("type") == "DOA_BEARING_UPDATE":
                    self.latest_state = {
                        "bearing_deg": float(data.get("compass_bearing_deg", 0.0)),
                        "spread_deg": float(data.get("compass_spread_deg", 12.0)),
                    }
        except Exception:
            pass
        finally:
            writer.close()

    async def start(self) -> None:
        self.server = await asyncio.start_server(self._handle_client, self.host, self.port)

    async def stop(self) -> None:
        if self.server:
            self.server.close()
            await self.server.wait_closed()
