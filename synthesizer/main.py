from synthesizer.llm_preprocessing import preprocessing
from synthesizer.query_production import query_production
from synthesizer.query_evolution import query_evolution
from openinference.instrumentation.langchain import LangChainInstrumentor

from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

tracer_provider = trace_sdk.TracerProvider()
span_exporter = OTLPSpanExporter("http://localhost:6006/v1/traces")
span_processor = SimpleSpanProcessor(span_exporter)
tracer_provider.add_span_processor(span_processor)
trace_api.set_tracer_provider(tracer_provider)

LangChainInstrumentor().instrument()


def main():
    preprocessing()
    query_production()
    query_evolution()

if __name__ == "__main__":
    main()