# bopomofo2hanzi
這裡存放我的研究：修正未切換注音輸入法產生之字元，中使用的程式碼。

## How to use?
### Host bot
1. 將 `example_config.json` 重新命名為 `config.json`
2. 在 `config.json` 中填入 Discord Bot Token
3. 執行以下指令安裝所需套件
    ```
    pip install -r requirements.txt
    ```
4. 執行 `bot.py`

### Train models
1. 將 `example_config.json` 重新命名為 `config.json`
2. 在 `config.json` 中填入模型資訊
3. 修改訓練程式中的 `model_name` 參數
3. 執行以下指令安裝所需套件
    ```
    pip install -r requirements.txt
    ```
4. 執行相對的訓練 Python 檔

## Datasets
TED: https://www.kaggle.com/datasets/wahabjawed/text-dataset-for-63-langauges  
PTT: https://github.com/zake7749/Gossiping-Chinese-Corpus
