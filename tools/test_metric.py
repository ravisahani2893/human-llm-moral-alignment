from app.metric_v1 import calculate_ccc

from app.metric_v1 import calculate_ccc


def test_perfect_match():
    human = [1, 2, 3, 4, 5]
    model = [1, 2, 3, 4, 5]

    ccc = calculate_ccc(human, model)

    assert round(ccc, 4) == 1.0000
    print("✅ Perfect Match Test Passed")


def test_similar_values():
    human = [-0.8, -0.2, 0.1, 0.5, 0.9]
    model = [-0.7, -0.1, 0.2, 0.6, 0.8]

    ccc = calculate_ccc(human, model)

    # We don't check the exact floating-point value
    assert 0.98 < ccc <= 1.0
    print("✅ Similar Values Test Passed")


def test_opposite_values():
    human = [1, 2, 3, 4, 5]
    model = [5, 4, 3, 2, 1]

    ccc = calculate_ccc(human, model)

    assert ccc < 0
    print("✅ Opposite Values Test Passed")


if __name__ == "__main__":
    test_perfect_match()
    test_similar_values()
    test_opposite_values()

    print("\n🎉 All CCC tests passed!")


