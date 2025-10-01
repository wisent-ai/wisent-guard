#1 class WisentSteeringTrainer:
#2 trainer should load activation collector (sse provded file).
#3 shoudl load contrastive pair set (see provided file)
#4 should decide what type of sterring trainig method choose: caa, bipo, etc.
#4 should be able to but from which layer we collect activation and then use for each activanis and layer steering method to obtain steering vector.
#5 some method uses many actviations from layers aome only one. we need to be able to specify that. like user can say use layer 10, 20, 30 or use all layers from 10 to 30.
#6 after training user need to obtain contrastive piars set with collected activatioons (see provded file) and steered vectors which need to be LayerActivations class.
#7 we should save all the trained sterred vectors, with contrastive pairs with activations, and meta data like date, model name, layers used, method used, hyperparams used etc.

# Imporatat info: we also need to sepcyfy activation collection stategy (see LayerActivations). All provded files has good descriptions/docstrings. Plse wrtire code with that in mind. create two files: atoms.py
# where we defied all base structure for the trainers, all abtarct calss etc. and steering_trainer.py where we implement WisentSteeringTrainer class.