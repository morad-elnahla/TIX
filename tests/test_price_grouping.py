import pytest
from agent.guards import prices_in, unverified_prices

@pytest.mark.parametrize("text", ["5000 جنيه", "5,000 جنيه", "5\u202f000 جنيه", "5\u00a0000 جنيه", "5 000 جنيه", "٥\u202f٠٠٠ جنيه"])
def test_grouped_price(text):
    assert prices_in(text) == [5000.0]
    assert unverified_prices(text, {5000}) == []
    assert unverified_prices(text, {4500}) == [5000.0]

def test_multiple_prices_remain_separate():
    assert prices_in("4500 جنيه و 10\u202f864 جنيه") == [4500, 10864]
