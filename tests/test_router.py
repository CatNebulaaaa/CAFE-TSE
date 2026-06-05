import torch

from cafe_tse.models.dynamic_router import DynamicRouter


def test_router_routes_and_active_blocks():
    router = DynamicRouter(0.35, 0.65)
    routes = router(torch.tensor([0.1, 0.5, 0.9]))
    assert routes == ["shallow", "lite", "full"]
    assert router.active_blocks(routes, 2, 3, 4) == [2, 3, 4]

