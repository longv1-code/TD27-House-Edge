"""House Edge training and offline prediction launcher."""
import os
import sys
sys.dont_write_bytecode=True
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
try:
    from next_model.runner import main
except ModuleNotFoundError as error:
    if error.name=='slot_forecasting':
        raise SystemExit('Put house_edge_next.py and next_model/ beside the kit public/ and slot_forecasting/ folders.')
    if error.name and error.name.startswith('next_model'):
        raise SystemExit('Extract the complete source ZIP, including its next_model/ folder.')
    raise SystemExit(f'Missing module: {error.name}. Run: python -m pip install -r requirements_next.txt')

if __name__=='__main__':
    main()
