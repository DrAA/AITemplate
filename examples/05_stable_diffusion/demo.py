#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import click
import torch

from aitemplate.testing.benchmark_pt import benchmark_torch_function
from pipeline_stable_diffusion_ait import StableDiffusionAITPipeline


@click.command()
@click.option("--token", default="", help="access token")
@click.option("--prompt", default="A vision of paradise, Unreal Engine", help="prompt")
@click.option("--num_images", default=10, help="Number of images to generate")
@click.option(
    "--benchmark", type=bool, default=False, help="run stable diffusion e2e benchmark"
)
def run(token, prompt, num_images, benchmark):
    pipe = StableDiffusionAITPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=token,
    ).to("cuda")

    with torch.autocast("cuda"):
        prompt_prefix = "".join((ch if ch.isalnum() else '-') for ch in prompt)
        for image_num in range(num_images):
            print(f'\nGenerating image {image_num}') 
            image = pipe(prompt).images[0]
            if benchmark:
                t = benchmark_torch_function(10, pipe, prompt)
                print(f"sd e2e: {t} ms")
            image.save(f"/output/{prompt_prefix}-{image_num}.png")


if __name__ == "__main__":
    run()
