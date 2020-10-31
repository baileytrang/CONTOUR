## CONTOUR: Penalty and Spotlight Mask for Abstractive Summarization

- **Authors**: Trang-Phuong N. Nguyen and Nhi-Thao Tran
- **Submitted** [ACIIDS 2021](aciids.pwr.edu.pl/2021)

The main purpose of **CONTOUR** is to emphasize the most related word containing the important information at each specific time by producing distinctive contours of the word’s potential. In contrast to baselines, which are [Pointer Generator Network](www.aclweb.org/anthology/P17-1099/) and [SELECTOR](www.aclweb.org/anthology/D19-1308/), we aim to take advantage of the last predicted word to the input words’ significance. 

The proposed Contour is an association of two independent sub-methods: **Penalty** and **Spotlight**.
- **Penalty** is the updated version of Selector that improves the focus areas generation and optimizes the inference time. 
- **Spotlight** generates the spotlight mask, which is used to re-ranking words potential by building an instance context from the last output word.

![image](https://user-images.githubusercontent.com/31720588/97787313-d4675500-1be3-11eb-8b20-cf450e940fd9.png)

![image](https://user-images.githubusercontent.com/31720588/97787305-c87b9300-1be3-11eb-995e-19e8cc4ce207.png)


## Datasets
We examined CONTOUR on multiple types of datasets and languages, which are:
- Large-scale [CNN/DailyMail](github.com/abisee/cnn-dailymail) for English
- Medium-scale [VNTC-Abs](github.com/trangnnp/VNTC-Abs) for Vietnamese
- Small-scale [Livedoor News Corpus](www.kaggle.com/vochicong/livedoor-news) for Japanese

## Result
| Dataset|R-1 | R-2 | R-L |
|----|---|---|---|
|**CNN/DailyMail** | 41.86 | 18.80 | 38.46 |
| **VNTC-Abs** | 26.80 | 9.50 | 24.16 |
| **Livedoor** | 31.68 | 14.83 | 29.26 | 

## How to run

1. Check the `data_util/config.py`, fill your path to store or load models
2. Prepare the dataset by converting it to chunked bin files by the `make_data_files.py`
3. **Train**: `python3 train.py`
4. **Evaluate**: `python3 eval.py --task=validate --model_name="model_name" --start_from=checkpoint.tar `
5. **Test**: `python3 eval.py --task=test --model_name="model_name" --load_model=checkpoint.tar `

**Citations**
```
@inproceedings{trangphuong-etal-2020-contour,
  author = {Trang-Phuong, N. Nguyen and Nhi-Thao, Tran},
  year = {2020},
  month = {10},
  title = {CONTOUR: Penalty and Spotlight Mask for Abstractive Summarization},
  publisher = "Submitted Asian Conference on Intelligent Information and Database Systems",
}
```
