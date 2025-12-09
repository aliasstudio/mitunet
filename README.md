# MitUNet

## References

* **Article**: [https://arxiv.org/html/2512.02413](https://arxiv.org/html/2512.02413)
* **Regional Dataset**: [https://doi.org/10.5281/zenodo.17871079](https://doi.org/10.5281/zenodo.17871079)
* **CubiCasa5k Dataset**:
    * Repo: [https://github.com/CubiCasa/CubiCasa5k](https://github.com/CubiCasa/CubiCasa5k)
    * Paper: [https://arxiv.org/pdf/1904.01920](https://arxiv.org/pdf/1904.01920)

## Project Structure

This repository is organized as follows:

* **`datasets/regional/`**: Contains the regional dataset.
* **`experiments/`**: Contains trained models from the article's research.
    * `experiments_mitunet.xlsx`: A spreadsheet with experiment results.
* **`images/`**: Contains images used in the article.
* **`notebooks/`**: Contains Jupyter notebooks.
    * **`MitUNet.ipynb`**: The main notebook used for training. Note that this notebook uses dataset versions hosted on Roboflow.

## Running Instructions

There are two primary ways to execute the code: using a local runtime with Docker (as performed in the study) or directly via Google Colab's cloud resources.

### Option 1: Local Runtime with Docker (Recommended)
To reproduce the exact environment and results from the article, we recommend using a **local runtime** connected to Google Colab. This allows the notebook to utilize your local hardware (GPU) via a Docker container.

1.  **Prerequisites**: Ensure you have Docker installed and a machine with a compatible GPU.
2.  **Setup Local Runtime**: Launch the Docker container on your machine to expose the Jupyter server.
    * For official instructions on connecting Colab to a local runtime, refer to: [Google Colab Local Runtimes Guide](https://research.google.com/colaboratory/local-runtimes.html).
3.  **Connect**: Open `notebooks/MitUNet.ipynb` in Google Colab. Click the arrow next to "Connect" (top right) and select **"Connect to a local runtime"**. Enter your local URL.

### Option 2: Google Colab (Cloud Runtime)
Alternatively, you can run the notebook directly on Google's cloud infrastructure if local hardware is unavailable.

1.  **Open**: Open `notebooks/MitUNet.ipynb` in Google Colab.
2.  **Select Runtime**: Go to `Runtime` > `Change runtime type` and select a GPU accelerator (e.g., T4, L4, or A100).
3.  **Resource Note**: Ensure the selected instance has sufficient VRAM and system RAM. If the free tier resources are insufficient for the batch size or model complexity, you may need a Colab Pro subscription or switch to Option 1.