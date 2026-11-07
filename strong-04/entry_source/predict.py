"""House Edge training and offline prediction launcher."""
import os
import sys
sys.dont_write_bytecode=True
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
try:
    from plus_model.runner import main
except ModuleNotFoundError as error:
    if error.name=='slot_forecasting':
        raise SystemExit('Put house_edge_plus.py and plus_model/ beside the kit public/ and slot_forecasting/ folders.')
    if error.name and error.name.startswith('plus_model'):
        raise SystemExit('Extract the complete source ZIP, including its plus_model/ folder.')
    raise SystemExit(f'Missing module: {error.name}. Run: python -m pip install -r requirements_plus.txt')

if __name__=='__main__':
    main()
