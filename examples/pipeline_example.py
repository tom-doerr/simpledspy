"""Pipeline examples for SimpleDSPy."""

import dspy

from simpledspy import ModuleFactory, PipelineManager, configure


configure(lm=dspy.LM(model="openai/gpt-3.5-turbo"), temperature=0.7, max_tokens=200)


def translation_pipeline_example():
    """Run text through a multi-step translation pipeline."""
    print("=== Translation Pipeline Example ===")

    factory = ModuleFactory()
    manager = PipelineManager()

    manager.register_step(
        inputs=["text"],
        outputs=["french_text"],
        module=factory.create_module(
            inputs=["text"],
            outputs=["french_text"],
            description="Translate the text to French",
        ),
    )
    manager.register_step(
        inputs=["french_text"],
        outputs=["spanish_text"],
        module=factory.create_module(
            inputs=["french_text"],
            outputs=["spanish_text"],
            description="Translate the French text to Spanish",
        ),
    )
    manager.register_step(
        inputs=["spanish_text"],
        outputs=["english_text"],
        module=factory.create_module(
            inputs=["spanish_text"],
            outputs=["english_text"],
            description="Translate the Spanish text back to English",
        ),
    )

    pipeline = manager.assemble_pipeline()
    original_text = "Hello, how are you today?"
    result = pipeline(text=original_text)

    print(f"Original: {original_text}")
    print(f"French: {result.french_text}")
    print(f"Spanish: {result.spanish_text}")
    print(f"Round trip: {result.english_text}\n")


def analysis_pipeline_example():
    """Analyze text with reasoning, summarization, and sentiment steps."""
    print("=== Text Analysis Pipeline ===")

    factory = ModuleFactory()
    manager = PipelineManager()

    manager.register_step(
        inputs=["text"],
        outputs=["key_points"],
        module=factory.create_module(
            inputs=["text"],
            outputs=["key_points"],
            description="Extract the main points from this text",
            module_type="chain_of_thought",
        ),
    )
    manager.register_step(
        inputs=["key_points"],
        outputs=["summary"],
        module=factory.create_module(
            inputs=["key_points"],
            outputs=["summary"],
            description="Create a concise summary from these key points",
        ),
    )
    manager.register_step(
        inputs=["summary"],
        outputs=["sentiment", "confidence"],
        module=factory.create_module(
            inputs=["summary"],
            outputs=["sentiment", "confidence"],
            description="Determine the overall sentiment and confidence level",
        ),
    )

    article = """
    The new environmental policy has received mixed reactions from stakeholders.
    Environmental groups praise the carbon reduction targets, while business
    leaders express concerns about costs and implementation timelines.
    """

    pipeline = manager.assemble_pipeline()
    result = pipeline(text=article)

    print(f"Original Article: {article.strip()}")
    print(f"Key Points: {result.key_points}")
    print(f"Summary: {result.summary}")
    print(f"Sentiment: {result.sentiment} (Confidence: {result.confidence})\n")


def data_processing_pipeline():
    """Parse, clean, and enrich raw business data."""
    print("=== Data Processing Pipeline ===")

    factory = ModuleFactory()
    manager = PipelineManager()

    manager.register_step(
        inputs=["raw_data"],
        outputs=["structured_data"],
        module=factory.create_module(
            inputs=["raw_data"],
            outputs=["structured_data"],
            description="Parse this raw data into a structured format",
        ),
    )
    manager.register_step(
        inputs=["structured_data"],
        outputs=["clean_data"],
        module=factory.create_module(
            inputs=["structured_data"],
            outputs=["clean_data"],
            description="Validate and clean the structured data",
        ),
    )
    manager.register_step(
        inputs=["clean_data"],
        outputs=["insights", "recommendations"],
        module=factory.create_module(
            inputs=["clean_data"],
            outputs=["insights", "recommendations"],
            description="Analyze the data and provide insights and recommendations",
            module_type="chain_of_thought",
        ),
    )

    raw_data = """
    Sales Q1: $1.2M (+15%), Q2: $1.1M (-8%), Q3: $1.3M (+18%), Q4: $1.5M (+15%)
    Top products: Widget A (35%), Gadget B (28%), Tool C (22%), Other (15%)
    Customer satisfaction: 4.2/5.0, Return rate: 3.2%
    """

    pipeline = manager.assemble_pipeline()
    result = pipeline(raw_data=raw_data)

    print(f"Raw Data: {raw_data.strip()}")
    print(f"Structured: {result.structured_data}")
    print(f"Clean: {result.clean_data}")
    print(f"Insights: {result.insights}")
    print(f"Recommendations: {result.recommendations}\n")


if __name__ == "__main__":
    translation_pipeline_example()
    analysis_pipeline_example()
    data_processing_pipeline()

    print("=== Pipeline examples completed! ===")
