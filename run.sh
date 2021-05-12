echo FOLD0
python main.py --verbose --fd 0 >> log.txt
cp -r new_trained_models/fold0/ pre_trained_models/
python main.py --test_mode --fd 0 >> log.txt
echo FOLD1
python main.py --verbose --fd 1 >> log.txt
cp -r new_trained_models/fold1/ pre_trained_models/
python main.py --test_mode --fd 1 >> log.txt
echo FOLD2
python main.py --verbose --fd 2 >> log.txt
cp -r new_trained_models/fold2/ pre_trained_models/
python main.py --test_mode --fd 2 >> log.txt
echo FOLD3
python main.py --verbose --fd 3 >> log.txt
cp -r new_trained_models/fold3/ pre_trained_models/
python main.py --test_mode --fd 3 >> log.txt
echo FOLD4
python main.py --verbose --fd 4 >> log.txt
cp -r new_trained_models/fold4/ pre_trained_models/
python main.py --test_mode --fd 4 >> log.txt
echo FOLD5
python main.py --verbose --fd 5 >> log.txt
cp -r new_trained_models/fold5/ pre_trained_models/
python main.py --test_mode --fd 5 >> log.txt
echo FOLD6
python main.py --verbose --fd 6 >> log.txt
cp -r new_trained_models/fold6/ pre_trained_models/
python main.py --test_mode --fd 6 >> log.txt
echo FOLD7
python main.py --verbose --fd 7 >> log.txt
cp -r new_trained_models/fold7/ pre_trained_models/
python main.py --test_mode --fd 7 >> log.txt
echo FOLD8
python main.py --verbose --fd 8 >> log.txt
cp -r new_trained_models/fold8/ pre_trained_models/
python main.py --test_mode --fd 8 >> log.txt
echo FOLD9
python main.py --verbose --fd 9 >> log.txt
cp -r new_trained_models/fold9/ pre_trained_models/
python main.py --test_mode --fd 9 >> log.txt
