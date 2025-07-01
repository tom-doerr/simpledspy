"""Pipeline example for SimpleDSPy - Multi-step processing"""

import dspy
from simpledspy import predict, chain_of_thought, configure, PipelineManager

# Configure language model
configure(lm=dspy.LM(model="openai/gpt-3.5-turbo"), temperature=0.7, max_tokens=200)


def translation_pipeline_example():
    """Example: Multi-language translation pipeline"""
    print("=== Translation Pipeline Example ===")

    # Create a pipeline manager
    pipeline = PipelineManager()

    # Step 1: Translate to French
    def translate_to_french(text: str) -> str:
        french_text = predict(text, outputs=["french_text"])
        return french_text

    # Step 2: Translate to Spanish
    def translate_to_spanish(french_text: str) -> str:
        spanish_text = predict(french_text, outputs=["spanish_text"])
        return spanish_text

    # Step 3: Translate back to English
    def translate_to_english(spanish_text: str) -> str:
        english_text = predict(spanish_text, outputs=["english_text"])
        return english_text

    # Add steps to pipeline
    pipeline.add_step("to_french", translate_to_french)
    pipeline.add_step("to_spanish", translate_to_spanish)
    pipeline.add_step("to_english", translate_to_english)

    # Run the pipeline
    original_text = "Hello, how are you today?"
    print(f"Original: {original_text}")

    result = pipeline.run(original_text)
    print(f"After round-trip translation: {result}")
    print()


def analysis_pipeline_example():
    """Example: Text analysis pipeline"""
    print("=== Text Analysis Pipeline ===")

    pipeline = PipelineManager()
    pipeline.reset()  # Clear any previous pipeline

    # Step 1: Extract key points
    def extract_key_points(text: str) -> str:
        key_points = chain_of_thought(
            text,
            outputs=["key_points"],
            description="Extract the main points from this text",
        )
        return key_points

    # Step 2: Generate summary
    def generate_summary(key_points: str) -> str:
        summary = predict(
            key_points,
            outputs=["summary"],
            description="Create a concise summary from these key points",
        )
        return summary

    # Step 3: Assess sentiment
    def assess_sentiment(summary: str) -> str:
        sentiment, confidence = predict(
            summary,
            outputs=["sentiment", "confidence"],
            description="Determine the overall sentiment and confidence level",
        )
        return f"Sentiment: {sentiment} (Confidence: {confidence})"

    # Build pipeline
    pipeline.add_step("extract", extract_key_points)
    pipeline.add_step("summarize", generate_summary)
    pipeline.add_step("sentiment", assess_sentiment)

    # Example text
    article = """
    The new environmental policy has received mixed reactions from various stakeholders.
    While environmental groups praise the ambitious targets for carbon reduction,
    business leaders express concerns about implementation costs and timeline feasibility.
    The policy aims to achieve net-zero emissions by 2040, requiring significant
    investments in renewable energy and infrastructure upgrades.
    """

    print(f"Original Article: {article.strip()}")
    print("\nPipeline Results:")
    final_result = pipeline.run(article)
    print(final_result)
    print()


def data_processing_pipeline():
    """Example: Data processing and enrichment pipeline"""
    print("=== Data Processing Pipeline ===")

    pipeline = PipelineManager()
    pipeline.reset()

    # Step 1: Parse and structure data
    def parse_data(raw_data: str) -> str:
        structured_data = predict(
            raw_data,
            outputs=["structured_data"],
            description="Parse this raw data into a structured format",
        )
        return structured_data

    # Step 2: Validate and clean
    def validate_clean(structured_data: str) -> str:
        clean_data = predict(
            structured_data,
            outputs=["clean_data"],
            description="Validate and clean the structured data",
        )
        return clean_data

    # Step 3: Enrich with analysis
    def enrich_data(clean_data: str) -> str:
        insights, recommendations = chain_of_thought(
            clean_data,
            outputs=["insights", "recommendations"],
            description="Analyze the data and provide insights and recommendations",
        )
        return f"Insights: {insights}\nRecommendations: {recommendations}"

    # Build pipeline
    pipeline.add_step("parse", parse_data)
    pipeline.add_step("clean", validate_clean)
    pipeline.add_step("enrich", enrich_data)

    # Example raw data
    raw_data = """
    Sales Q1: $1.2M (↑15%), Q2: $1.1M (↓8%), Q3: $1.3M (↑18%), Q4: $1.5M (↑15%)
    Top products: Widget A (35%), Gadget B (28%), Tool C (22%), Other (15%)
    Customer satisfaction: 4.2/5.0, Return rate: 3.2%
    """

    print(f"Raw Data: {raw_data.strip()}")
    print("\nProcessed Results:")
    results = pipeline.run(raw_data)
    print(results)
    print()


def get_pipeline_assembly():
    """Show how to inspect pipeline assembly"""
    print("=== Pipeline Assembly Inspection ===")

    pipeline = PipelineManager()
    pipeline.reset()

    # Add some example steps
    pipeline.add_step("step1", lambda x: predict(x, outputs=["output1"]))
    pipeline.add_step("step2", lambda output1: predict(output1, outputs=["output2"]))
    pipeline.add_step("step3", lambda output2: predict(output2, outputs=["final"]))

    # Get pipeline assembly as a DSPy module
    assembled_pipeline = pipeline.get_pipeline_module()
    print(f"Pipeline assembled: {assembled_pipeline}")
    print(f"Number of steps: {len(pipeline.steps)}")
    print()


if __name__ == "__main__":
    # Run all pipeline examples
    translation_pipeline_example()
    analysis_pipeline_example()
    data_processing_pipeline()
    get_pipeline_assembly()

    print("=== Pipeline examples completed! ===")
    print("Pipelines are great for complex, multi-step processing tasks.")
