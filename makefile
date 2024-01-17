train:
	python -m src.defect_classification.train

basic-benchmark:
	python -m src.defect_classification.basic_benchmark

fp16-benchmark:
	python -m src.defect_classification.fp16_benchmark

int8-benchmark:
	python -m src.defect_classification.int8_benchmark

serve:
	uvicorn server.server:app --reload