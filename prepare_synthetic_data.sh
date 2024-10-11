python grammar_gnerate_prompt.py
sh cap_synthic_code/train_text_cls.sh
cp -rf cap_synthic_code/ScanRefer_filtered_train.json DATA_ROOT/scanrefer/
mv -f scanrefer_pred_spans_train.json DATA_ROOT/ 
