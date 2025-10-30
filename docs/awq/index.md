# AutoAWQ

AutoAWQ pushes ease of use and fast inference speed into one package. In the following documentation,
you will learn how to quantize and run inference.

Example inference speed (RTX 4090, Ryzen 9 7950X, 64 tokens):

- Vicuna 7B (GEMV kernel): 198.848 tokens/s
- Mistral 7B (GEMM kernel): 156.317 tokens/s
- Mistral 7B (ExLlamaV2 kernel): 188.865 tokens/s
- Mixtral 46.7B (GEMM kernel): 93 tokens/s (2x 4090)

> ⚠️ Warning: The AutoAWQ library is deprecated. This functionality has been adopted by the vLLM project in llm-compressor. For the recommended quantization workflow, please see the AWQ examples in llm-compressor. For more details on the deprecation, refer to the original AutoAWQ repository.

To create a new 4-bit quantized model, you can leverage AutoAWQ. Quantization reduces the model's precision from BF16/FP16 to INT4, lowering memory footprint and improving latency.

You can quantize your own models by installing AutoAWQ or by using one of the many models on Hugging Face.

<!-- AWQ docs removed during migration to llm-compressor. -->
This doc was removed during a migration away from AutoAWQ. The project now recommends using the
llm-compressor tool maintained by the vLLM project for reliable AWQ-style quantization. See the
project README and TODO.md for the migration plan.
# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')
```

To run an AWQ model with vLLM (offline example):

```bash
python examples/offline_inference/llm_engine_example.py \
    --model TheBloke/Llama-2-7b-Chat-AWQ \
    --quantization awq
```

AWQ models are also supported directly through vLLM's LLM entrypoint:

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="TheBloke/Llama-2-7b-Chat-AWQ", quantization="AWQ")
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

For more details and advanced options, refer to the AutoAWQ and vLLM documentation and the AWQ examples in llm-compressor.

## Installation notes

- Install: `pip install autoawq`.
- Your torch version must match the build version, i.e. you cannot use torch 2.0.1 with a wheel that was built with 2.2.0.
- For AMD GPUs, inference will run through ExLlamaV2 kernels without fused layers. You need to pass the following arguments to run with AMD GPUs:
    ```python
    model = AutoAWQForCausalLM.from_quantized(
        ...,
        fuse_layers=False,
        use_exllama_v2=True
    )
    ```
- For CPU device, you should install intel_extension_for_pytorch with `pip install intel_extension_for_pytorch`. And the latest version of torch is required since "intel_extension_for_pytorch(IPEX)" was built with the latest version of torch(now IPEX 2.4 was build with torch 2.4). If you build IPEX from source code, then you need to ensure the consistency of the torch version. And you should use "use_ipex=True" for CPU device.
    ```python
    model = AutoAWQForCausalLM.from_quantized(
        ...,
        use_ipex=True
    )
    ```

## Supported models

We support modern LLMs. You can find a list of supported Huggingface `model_types` in `awq/models`.
