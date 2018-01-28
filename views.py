from aiohttp import web
from test_runner import test

async def index(request):
    return web.Response(text="Test in progress")
