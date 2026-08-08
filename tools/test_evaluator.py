from app.evaluator import evaluate_alignment
from tools.test_metric import test_similar_values 

def test_evaluate_alignment():
    
    model_name = "gemini"

    result = evaluate_alignment(
        model_name
    )

    print(f"Result from evaluate_alignment gemini: {result}")

    model_name = "claude"
    
    result = evaluate_alignment(
            model_name
    )
    
    print(f"Result from evaluate_alignment claude: {result}")
    print("✅ evaluate_alignment Test Passed")


if __name__ == "__main__":
    test_evaluate_alignment()