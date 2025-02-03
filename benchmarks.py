from Curate import Curate
from rerank import rankers
import json

example_queries = [
    "Learning Python",
    "Python for Data Science",
    "Infrastructure as Code for Cloud Architectures",
    "Enterprise Resource Planning Foundations",
    "Introduction to Machine Learning",
    "Sales Management",
]

if __name__ == "__main__":
    results = {}
    for index, model_name in enumerate(rankers.keys()):
        print(f"Model {index+1} of {len(rankers)}: {model_name}")
        try:
            for query in example_queries:
                print(f"Query: {query}")
                options = Curate(query, model_name=model_name)
                results[model_name] = options
        except Exception as e:
            error_message = f"Error: {e}"
            print(error_message)
            results[model_name] = error_message
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
