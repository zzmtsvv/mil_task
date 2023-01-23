from utils import seed_everything
from trainers import AutoEncoderTrainer, ClassifierTrainer

seed_everything()
#ae = AutoEncoderTrainer()
#ae.fit()
t = ClassifierTrainer()
t.fit()
