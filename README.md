# ugip_telework-avatar

## work-flow
1. making dataset and generating model
  ```bash
  $ python setup_main.py
  ```
  - collect pictures of face taken periodically
  - process pictures into several numerical values using face++
  - set training data in terms of emotions/facial features  
  - regression analysis / classification / neural network  

2. estimating your stress 
```bash
$ python calc_tiredness.py
```

3. transfer data(less important)
  - socket(TCP/IP) communication  


4. visualize tiredness/current working condition
  - sticky notes(windows)
  - tkinter(python module)
  - modify desktop goose(if possible)
