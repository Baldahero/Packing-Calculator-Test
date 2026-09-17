# Packing Calculator - Test Version

Safe test copy of the NorDan packing calculator. The original repository is not modified.

## Confirmed scope

- Only 1200 mm deep pallets and glass boxes are calculated.
- Ireland pallet and glass-box weight limit is 1000 kg.
- uPVC products are outside the scope.
- Facades are packed by weight only, without a unit-count limit.
- LDM for a 1200 mm pallet is `actual pallet length / 2000`.
- Constructions higher than 2700 mm are packed sideways and use height + 200 mm as pallet length.
- The entered glass weight is total glass weight per construction and is not multiplied by leaf/part count.

## Conservative assumptions

The supplied packing sheet ties window capacities (6/8/10) to pictured configurations. The current input form does not contain enough information to identify those pictures. Until a configuration selector is added, windows use a conservative maximum of 6 units on a 1200 mm pallet.

## Run

```bash
pip install -r requirements.txt
streamlit run PK.py
```

## Test

```bash
python -m unittest discover -s tests -v
```
