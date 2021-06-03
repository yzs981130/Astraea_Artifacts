# simulation

# requirements
```
python >= 3.5
```


# run
```
python main.py # change arguments in options.py
```
# results
|  scheduler   | avgJCT  | MissDDL |
|  ----   | ----  | ----  |
|  FIFO   | 17530  | 0 |
|  DLAS   | 17608  | 0  |
|  Gitt   | 16489 | 0 |
| Gandiva | 27238 | 0 |
|Time-Aware| 16489  | 0 |


# data generation
```
cd data/; python trace_generator.py
```