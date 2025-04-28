import llm_executor
import llm_judge
import llm_planner
import setup

def main():
    # If local model is used s1 and s2 are api_base and model
    api_used, model_used, config_path, history_path = setup.initial_setup()

    llm_planner.engage_llm(api_used, model_used, config_path, history_path)

if __name__ == "__main__":
    main()