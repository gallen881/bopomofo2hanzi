# bopomofo2hanzi
這裡存放我的研究：修正未切換注音輸入法產生之字元，中使用的程式碼。

## How to use?
### Host Discord bot
1. 將 `config_example.json` 重新命名為 `config.json`
2. 在 `config.json` 中填入 Discord Bot Token
3. 執行以下指令安裝所需套件
    ```
    pip install -r requirements.txt
    ```
4. 執行 `bot.py`

### Train models
依自身情況修改 `train_XXX.py` 中 `get_datasets_and_tv` 的參數以及其他模型參數，並執行 `train_XXX.py`

### Use model in terminal
1. 確認模型已訓練好且位於正確的目錄下
2. 依自身情況修改 `translator_XXX.py` 中 `model` 的值，並執行 `translator_XXX.py`
> RNN 以及 LSTM 模型共用 `translator_RNN.py`，修改 `model` 值即可  
> mT5 模型相關檔案為 Jupyter Notebook

### Online translator
https://gallen881.github.io/bopomofo2hanzi/

## Datasets
TED: https://www.kaggle.com/datasets/wahabjawed/text-dataset-for-63-langauges  
PTT: https://github.com/zake7749/Gossiping-Chinese-Corpus
