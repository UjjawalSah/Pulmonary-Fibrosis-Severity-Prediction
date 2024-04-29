{
  "metadata": {
    "kernelspec": {
      "language": "python",
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "pygments_lexer": "ipython3",
      "nbconvert_exporter": "python",
      "version": "3.6.4",
      "file_extension": ".py",
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "name": "python",
      "mimetype": "text/x-python"
    },
    "kaggle": {
      "accelerator": "gpu",
      "dataSources": [
        {
          "sourceId": 20604,
          "databundleVersionId": 1357052,
          "sourceType": "competition"
        }
      ],
      "dockerImageVersionId": 29981,
      "isInternetEnabled": false,
      "language": "python",
      "sourceType": "notebook",
      "isGpuEnabled": true
    },
    "colab": {
      "name": "Linear Decay (based on ResNet CNN)",
      "provenance": [],
      "include_colab_link": true
    }
  },
  "nbformat_minor": 0,
  "nbformat": 4,
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/UjjawalSah/Predicting-the-lung-function/blob/main/Linear_Decay_(based_on_ResNet_CNN).py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "source": [
        "\n",
        "# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES\n",
        "# TO THE CORRECT LOCATION (/kaggle/input) IN YOUR NOTEBOOK,\n",
        "# THEN FEEL FREE TO DELETE THIS CELL.\n",
        "# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON\n",
        "# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR\n",
        "# NOTEBOOK.\n",
        "\n",
        "import os\n",
        "import sys\n",
        "from tempfile import NamedTemporaryFile\n",
        "from urllib.request import urlopen\n",
        "from urllib.parse import unquote, urlparse\n",
        "from urllib.error import HTTPError\n",
        "from zipfile import ZipFile\n",
        "import tarfile\n",
        "import shutil\n",
        "\n",
        "CHUNK_SIZE = 40960\n",
        "DATA_SOURCE_MAPPING = 'osic-pulmonary-fibrosis-progression:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-competitions-data%2Fkaggle-v2%2F20604%2F1357052%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240429%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240429T114836Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D6e19247837a88b0813b6f70e63d1b9e136dec2c9257284ef9a9e89e9c0ab89f6ad60f951e60c50c959f40507de0a77e6c19ef63feff15042b3cba4a91e4808b9ba0e2191fbd7bc1ef64b640c96d983aa00ea39adec82ad223a292fc19f01b73f8ab765a0258be06164e8765b67c10b37d7f0305c5f9e7260f2a99470ecf1146f0d420ff0ad17afadae53ae42b0eef7bf7220e8e3f64bb7d1e7e21748a3b58f07e426932d9e9276e74ef6130f77891e2e71eb42527172a17135590d448b0979d2dfd31f7e1f1b5411f96f0d7ad6c95af303bf1f034a205a1fae2c914639ad2ed341d1133556c0cbdaf43af91316278be8850167d42dcda8f41cc88e55b0bd00d2'\n",
        "\n",
        "KAGGLE_INPUT_PATH='/kaggle/input'\n",
        "KAGGLE_WORKING_PATH='/kaggle/working'\n",
        "KAGGLE_SYMLINK='kaggle'\n",
        "\n",
        "!umount /kaggle/input/ 2> /dev/null\n",
        "shutil.rmtree('/kaggle/input', ignore_errors=True)\n",
        "os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)\n",
        "os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)\n",
        "\n",
        "try:\n",
        "  os.symlink(KAGGLE_INPUT_PATH, os.path.join(\"..\", 'input'), target_is_directory=True)\n",
        "except FileExistsError:\n",
        "  pass\n",
        "try:\n",
        "  os.symlink(KAGGLE_WORKING_PATH, os.path.join(\"..\", 'working'), target_is_directory=True)\n",
        "except FileExistsError:\n",
        "  pass\n",
        "\n",
        "for data_source_mapping in DATA_SOURCE_MAPPING.split(','):\n",
        "    directory, download_url_encoded = data_source_mapping.split(':')\n",
        "    download_url = unquote(download_url_encoded)\n",
        "    filename = urlparse(download_url).path\n",
        "    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)\n",
        "    try:\n",
        "        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:\n",
        "            total_length = fileres.headers['content-length']\n",
        "            print(f'Downloading {directory}, {total_length} bytes compressed')\n",
        "            dl = 0\n",
        "            data = fileres.read(CHUNK_SIZE)\n",
        "            while len(data) > 0:\n",
        "                dl += len(data)\n",
        "                tfile.write(data)\n",
        "                done = int(50 * dl / int(total_length))\n",
        "                sys.stdout.write(f\"\\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded\")\n",
        "                sys.stdout.flush()\n",
        "                data = fileres.read(CHUNK_SIZE)\n",
        "            if filename.endswith('.zip'):\n",
        "              with ZipFile(tfile) as zfile:\n",
        "                zfile.extractall(destination_path)\n",
        "            else:\n",
        "              with tarfile.open(tfile.name) as tarfile:\n",
        "                tarfile.extractall(destination_path)\n",
        "            print(f'\\nDownloaded and uncompressed: {directory}')\n",
        "    except HTTPError as e:\n",
        "        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')\n",
        "        continue\n",
        "    except OSError as e:\n",
        "        print(f'Failed to load {download_url} to path {destination_path}')\n",
        "        continue\n",
        "\n",
        "print('Data source import complete.')\n"
      ],
      "metadata": {
        "id": "Rdilkvdcr5Fk",
        "outputId": "d3353002-7054-408d-8866-1762255bdcff",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "cell_type": "code",
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Downloading osic-pulmonary-fibrosis-progression, 14332865109 bytes compressed\n",
            "[==                                                ] 575119360 bytes downloaded"
          ]
        }
      ],
      "execution_count": null
    },
    {
      "cell_type": "code",
      "source": [
        "pip install pydicom"
      ],
      "metadata": {
        "id": "v1I3tZHFsVkr",
        "outputId": "2af2ad9a-d6ae-442b-a726-77d1f99c7194",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting pydicom\n",
            "  Downloading pydicom-2.4.4-py3-none-any.whl (1.8 MB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m1.8/1.8 MB\u001b[0m \u001b[31m10.8 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hInstalling collected packages: pydicom\n",
            "Successfully installed pydicom-2.4.4\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "import cv2\n",
        "\n",
        "import pydicom\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import tensorflow as tf\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "from tqdm.notebook import tqdm"
      ],
      "metadata": {
        "_cell_guid": "79c7e3d0-c299-4dcb-8224-4455121ee9b0",
        "_uuid": "d629ff2d2480ee46fbb7e2d37f6b5fab8052498a",
        "trusted": true,
        "id": "r4A80zV3r5Fo"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Decay theory\n",
        "Input for test:\n",
        "   * FVC in n week\n",
        "   * Percent in n week\n",
        "   * Age\n",
        "   * Sex\n",
        "   * Smoking status\n",
        "   * CT in n week\n",
        "   \n",
        "Result:\n",
        "   * FVC in any week\n",
        "   * percent in any week\n",
        "   \n",
        "$FVC = a.quantile(0.75) * (week - week_{test}) + FVC_{test}$\n",
        "\n",
        "$Confidence = Percent + a.quantile(0.75) * abs(week - week_{test}) $\n",
        "\n",
        "So let's try predict coefficient a."
      ],
      "metadata": {
        "id": "qt2zjyjpr5Fp"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')"
      ],
      "metadata": {
        "trusted": true,
        "id": "-1MD6R0Yr5Fs",
        "outputId": "aa5d4534-6f29-41cb-afe4-e523a5f47eeb",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 287
        }
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "error",
          "ename": "FileNotFoundError",
          "evalue": "[Errno 2] No such file or directory: '../input/osic-pulmonary-fibrosis-progression/train.csv'",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mFileNotFoundError\u001b[0m                         Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-4-93079757735d>\u001b[0m in \u001b[0;36m<cell line: 1>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mtrain\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mpd\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mread_csv\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'../input/osic-pulmonary-fibrosis-progression/train.csv'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/pandas/io/parsers/readers.py\u001b[0m in \u001b[0;36mread_csv\u001b[0;34m(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)\u001b[0m\n\u001b[1;32m    910\u001b[0m     \u001b[0mkwds\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mupdate\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mkwds_defaults\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    911\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 912\u001b[0;31m     \u001b[0;32mreturn\u001b[0m \u001b[0m_read\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfilepath_or_buffer\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mkwds\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    913\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    914\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/pandas/io/parsers/readers.py\u001b[0m in \u001b[0;36m_read\u001b[0;34m(filepath_or_buffer, kwds)\u001b[0m\n\u001b[1;32m    575\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    576\u001b[0m     \u001b[0;31m# Create the parser.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 577\u001b[0;31m     \u001b[0mparser\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mTextFileReader\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfilepath_or_buffer\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwds\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    578\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    579\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mchunksize\u001b[0m \u001b[0;32mor\u001b[0m \u001b[0miterator\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/pandas/io/parsers/readers.py\u001b[0m in \u001b[0;36m__init__\u001b[0;34m(self, f, engine, **kwds)\u001b[0m\n\u001b[1;32m   1405\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1406\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mhandles\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mIOHandles\u001b[0m \u001b[0;34m|\u001b[0m \u001b[0;32mNone\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1407\u001b[0;31m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_engine\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_make_engine\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mf\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mengine\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1408\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1409\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mclose\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m->\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/pandas/io/parsers/readers.py\u001b[0m in \u001b[0;36m_make_engine\u001b[0;34m(self, f, engine)\u001b[0m\n\u001b[1;32m   1659\u001b[0m                 \u001b[0;32mif\u001b[0m \u001b[0;34m\"b\"\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mmode\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1660\u001b[0m                     \u001b[0mmode\u001b[0m \u001b[0;34m+=\u001b[0m \u001b[0;34m\"b\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1661\u001b[0;31m             self.handles = get_handle(\n\u001b[0m\u001b[1;32m   1662\u001b[0m                 \u001b[0mf\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1663\u001b[0m                 \u001b[0mmode\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/pandas/io/common.py\u001b[0m in \u001b[0;36mget_handle\u001b[0;34m(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)\u001b[0m\n\u001b[1;32m    857\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mioargs\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mencoding\u001b[0m \u001b[0;32mand\u001b[0m \u001b[0;34m\"b\"\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mioargs\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmode\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    858\u001b[0m             \u001b[0;31m# Encoding\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 859\u001b[0;31m             handle = open(\n\u001b[0m\u001b[1;32m    860\u001b[0m                 \u001b[0mhandle\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    861\u001b[0m                 \u001b[0mioargs\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmode\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mFileNotFoundError\u001b[0m: [Errno 2] No such file or directory: '../input/osic-pulmonary-fibrosis-progression/train.csv'"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "train.head()"
      ],
      "metadata": {
        "trusted": true,
        "id": "Eu_6ju8Kr5Ft"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train.SmokingStatus.unique()"
      ],
      "metadata": {
        "trusted": true,
        "id": "hV2ZFlaVr5Ft"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def get_tab(df):\n",
        "    vector = [(df.Age.values[0] - 30) / 30]\n",
        "\n",
        "    if df.Sex.values[0] == 'male':\n",
        "       vector.append(0)\n",
        "    else:\n",
        "       vector.append(1)\n",
        "\n",
        "    if df.SmokingStatus.values[0] == 'Never smoked':\n",
        "        vector.extend([0,0])\n",
        "    elif df.SmokingStatus.values[0] == 'Ex-smoker':\n",
        "        vector.extend([1,1])\n",
        "    elif df.SmokingStatus.values[0] == 'Currently smokes':\n",
        "        vector.extend([0,1])\n",
        "    else:\n",
        "        vector.extend([1,0])\n",
        "    return np.array(vector)"
      ],
      "metadata": {
        "trusted": true,
        "id": "N1GoWyuhr5Ft"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "A = {}\n",
        "TAB = {}\n",
        "P = []\n",
        "for i, p in tqdm(enumerate(train.Patient.unique())):\n",
        "    sub = train.loc[train.Patient == p, :]\n",
        "    fvc = sub.FVC.values\n",
        "    weeks = sub.Weeks.values\n",
        "    c = np.vstack([weeks, np.ones(len(weeks))]).T\n",
        "    a, b = np.linalg.lstsq(c, fvc)[0]\n",
        "\n",
        "    A[p] = a\n",
        "    TAB[p] = get_tab(sub)\n",
        "    P.append(p)"
      ],
      "metadata": {
        "trusted": true,
        "id": "xCIr7Lmir5Fu"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## CNN for coeff prediction"
      ],
      "metadata": {
        "id": "aOJJwemhr5Fv"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def get_img(path):\n",
        "    d = pydicom.dcmread(path)\n",
        "    return cv2.resize(d.pixel_array / 2**11, (512, 512))"
      ],
      "metadata": {
        "trusted": true,
        "id": "8qV-9fMQr5Fv"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from tensorflow.keras.utils import Sequence\n",
        "\n",
        "class IGenerator(Sequence):\n",
        "    BAD_ID = ['ID00011637202177653955184', 'ID00052637202186188008618']\n",
        "    def __init__(self, keys, a, tab, batch_size=32):\n",
        "        self.keys = [k for k in keys if k not in self.BAD_ID]\n",
        "        self.a = a\n",
        "        self.tab = tab\n",
        "        self.batch_size = batch_size\n",
        "\n",
        "        self.train_data = {}\n",
        "        for p in train.Patient.values:\n",
        "            self.train_data[p] = os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{p}/')\n",
        "\n",
        "    def __len__(self):\n",
        "        return 1000\n",
        "\n",
        "    def __getitem__(self, idx):\n",
        "        x = []\n",
        "        a, tab = [], []\n",
        "        keys = np.random.choice(self.keys, size = self.batch_size)\n",
        "        for k in keys:\n",
        "            try:\n",
        "                i = np.random.choice(self.train_data[k], size=1)[0]\n",
        "                img = get_img(f'../input/osic-pulmonary-fibrosis-progression/train/{k}/{i}')\n",
        "                x.append(img)\n",
        "                a.append(self.a[k])\n",
        "                tab.append(self.tab[k])\n",
        "            except:\n",
        "                print(k, i)\n",
        "\n",
        "        x,a,tab = np.array(x), np.array(a), np.array(tab)\n",
        "        x = np.expand_dims(x, axis=-1)\n",
        "        return [x, tab] , a"
      ],
      "metadata": {
        "trusted": true,
        "id": "WBNct1B5r5Fw"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from tensorflow.keras.layers import (\n",
        "    Dense, Dropout, Activation, Flatten, Input, BatchNormalization, GlobalAveragePooling2D, Add, Conv2D, AveragePooling2D,\n",
        "    LeakyReLU, Concatenate\n",
        ")\n",
        "\n",
        "from tensorflow.keras import Model\n",
        "from tensorflow.keras.optimizers import Nadam\n",
        "\n",
        "def get_model(shape=(512, 512, 1)):\n",
        "    def res_block(x, n_features):\n",
        "        _x = x\n",
        "        x = BatchNormalization()(x)\n",
        "        x = LeakyReLU(0.05)(x)\n",
        "\n",
        "        x = Conv2D(n_features, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)\n",
        "        x = Add()([_x, x])\n",
        "        return x\n",
        "\n",
        "    inp = Input(shape=shape)\n",
        "\n",
        "    # 512\n",
        "    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(inp)\n",
        "    x = BatchNormalization()(x)\n",
        "    x = LeakyReLU(0.05)(x)\n",
        "\n",
        "    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)\n",
        "    x = BatchNormalization()(x)\n",
        "    x = LeakyReLU(0.05)(x)\n",
        "\n",
        "    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)\n",
        "\n",
        "    # 256\n",
        "    x = Conv2D(8, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)\n",
        "    for _ in range(2):\n",
        "        x = res_block(x, 8)\n",
        "    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)\n",
        "\n",
        "    # 128\n",
        "    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)\n",
        "    for _ in range(2):\n",
        "        x = res_block(x, 16)\n",
        "    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)\n",
        "\n",
        "    # 64\n",
        "    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)\n",
        "    for _ in range(3):\n",
        "        x = res_block(x, 32)\n",
        "    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)\n",
        "\n",
        "    # 32\n",
        "    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)\n",
        "    for _ in range(3):\n",
        "        x = res_block(x, 64)\n",
        "    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)\n",
        "\n",
        "    # 16\n",
        "    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)\n",
        "    for _ in range(3):\n",
        "        x = res_block(x, 128)\n",
        "\n",
        "    # 16\n",
        "    x = GlobalAveragePooling2D()(x)\n",
        "\n",
        "    inp2 = Input(shape=(4,))\n",
        "    x2 = tf.keras.layers.GaussianNoise(0.2)(inp2)\n",
        "    x = Concatenate()([x, x2])\n",
        "    x = Dropout(0.6)(x)\n",
        "    x = Dense(1)(x)\n",
        "    #x2 = Dense(1)(x)\n",
        "    return Model([inp, inp2] , x)"
      ],
      "metadata": {
        "trusted": true,
        "id": "VaNnLehpr5Fx"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model = get_model()\n",
        "model.summary()"
      ],
      "metadata": {
        "trusted": true,
        "id": "A0QrK_58r5Fx"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from tensorflow_addons.optimizers import RectifiedAdam\n",
        "\n",
        "model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mae')"
      ],
      "metadata": {
        "trusted": true,
        "id": "QoN2LWdUr5Fy"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "tr_p, vl_p = train_test_split(P,\n",
        "                              shuffle=True,\n",
        "                              train_size= 0.8)"
      ],
      "metadata": {
        "trusted": true,
        "id": "q83CnLorr5Fy"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import seaborn as sns\n",
        "\n",
        "sns.distplot(list(A.values()));"
      ],
      "metadata": {
        "trusted": true,
        "id": "2_tlzZz3r5Fz"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "er = tf.keras.callbacks.EarlyStopping(\n",
        "    monitor=\"val_loss\",\n",
        "    min_delta=1e-3,\n",
        "    patience=5,\n",
        "    verbose=0,\n",
        "    mode=\"auto\",\n",
        "    baseline=None,\n",
        "    restore_best_weights=True,\n",
        ")"
      ],
      "metadata": {
        "trusted": true,
        "id": "ZLvaAnXDr5Fz"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model.fit_generator(IGenerator(keys=tr_p,\n",
        "                               a = A,\n",
        "                               tab = TAB),\n",
        "                    steps_per_epoch = 200,\n",
        "                    validation_data=IGenerator(keys=vl_p,\n",
        "                               a = A,\n",
        "                               tab = TAB),\n",
        "                    validation_steps = 20,\n",
        "                    callbacks = [er],\n",
        "                    epochs=30)"
      ],
      "metadata": {
        "trusted": true,
        "id": "q1JcQBdBr5F0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def score(fvc_true, fvc_pred, sigma):\n",
        "    sigma_clip = np.maximum(sigma, 70)\n",
        "    delta = np.abs(fvc_true - fvc_pred)\n",
        "    delta = np.minimum(delta, 1000)\n",
        "    sq2 = np.sqrt(2)\n",
        "    metric = (delta / sigma_clip)*sq2 + np.log(sigma_clip* sq2)\n",
        "    return np.mean(metric)"
      ],
      "metadata": {
        "trusted": true,
        "id": "ggSj2tHCr5F1"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from tqdm.notebook import tqdm\n",
        "\n",
        "metric = []\n",
        "for q in tqdm(range(1, 10)):\n",
        "    m = []\n",
        "    for p in vl_p:\n",
        "        x = []\n",
        "        tab = []\n",
        "\n",
        "        if p in ['ID00011637202177653955184', 'ID00052637202186188008618']:\n",
        "            continue\n",
        "        for i in os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{p}/'):\n",
        "            x.append(get_img(f'../input/osic-pulmonary-fibrosis-progression/train/{p}/{i}'))\n",
        "            tab.append(get_tab(train.loc[train.Patient == p, :]))\n",
        "        tab = np.array(tab)\n",
        "\n",
        "        x = np.expand_dims(x, axis=-1)\n",
        "        _a = model.predict([x, tab])\n",
        "        a = np.quantile(_a, q / 10)\n",
        "\n",
        "        percent_true = train.Percent.values[train.Patient == p]\n",
        "        fvc_true = train.FVC.values[train.Patient == p]\n",
        "        weeks_true = train.Weeks.values[train.Patient == p]\n",
        "\n",
        "        fvc = a * (weeks_true - weeks_true[0]) + fvc_true[0]\n",
        "        percent = percent_true[0] - a * abs(weeks_true - weeks_true[0])\n",
        "        m.append(score(fvc_true, fvc, percent))\n",
        "    print(np.mean(m))\n",
        "    metric.append(np.mean(m))"
      ],
      "metadata": {
        "trusted": true,
        "id": "8ygznzHxr5F1"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Predict"
      ],
      "metadata": {
        "id": "9U3xmXn4r5F2"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "q = (np.argmin(metric) + 1)/ 10\n",
        "q"
      ],
      "metadata": {
        "trusted": true,
        "id": "mgW9tzCyr5F2"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sub = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')\n",
        "sub.head()"
      ],
      "metadata": {
        "trusted": true,
        "id": "EboueWtwr5F3"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')\n",
        "test.head()"
      ],
      "metadata": {
        "trusted": true,
        "id": "kPFas7f3r5F3"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "A_test, B_test, P_test,W, FVC= {}, {}, {},{},{}\n",
        "STD, WEEK = {}, {}\n",
        "for p in test.Patient.unique():\n",
        "    x = []\n",
        "    tab = []\n",
        "    for i in os.listdir(f'../input/osic-pulmonary-fibrosis-progression/test/{p}/'):\n",
        "        x.append(get_img(f'../input/osic-pulmonary-fibrosis-progression/test/{p}/{i}'))\n",
        "        tab.append(get_tab(test.loc[test.Patient == p, :]))\n",
        "    tab = np.array(tab)\n",
        "\n",
        "    x = np.expand_dims(x, axis=-1)\n",
        "    _a = model.predict([x, tab])\n",
        "    a = np.quantile(_a, q)\n",
        "    A_test[p] = a\n",
        "    B_test[p] = test.FVC.values[test.Patient == p] - a*test.Weeks.values[test.Patient == p]\n",
        "    P_test[p] = test.Percent.values[test.Patient == p]\n",
        "    WEEK[p] = test.Weeks.values[test.Patient == p]"
      ],
      "metadata": {
        "trusted": true,
        "id": "NGWFsNJEr5F3"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "for k in sub.Patient_Week.values:\n",
        "    p, w = k.split('_')\n",
        "    w = int(w)\n",
        "\n",
        "    fvc = A_test[p] * w + B_test[p]\n",
        "    sub.loc[sub.Patient_Week == k, 'FVC'] = fvc\n",
        "    sub.loc[sub.Patient_Week == k, 'Confidence'] = (\n",
        "        P_test[p] - A_test[p] * abs(WEEK[p] - w)\n",
        ")\n",
        ""
      ],
      "metadata": {
        "trusted": true,
        "id": "f8GREifRr5F3"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sub.head()"
      ],
      "metadata": {
        "trusted": true,
        "id": "N9jdqw8Sr5F4"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "sub[[\"Patient_Week\",\"FVC\",\"Confidence\"]].to_csv(\"submission.csv\", index=False)"
      ],
      "metadata": {
        "trusted": true,
        "id": "H-ev-AVIr5F4"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "trusted": true,
        "id": "oh5S_NDHr5F4"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "trusted": true,
        "id": "EFRvX_jLr5F4"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}