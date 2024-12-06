# CS229Project
Code for CS229 Group Project

Run TrainCritic.py

Those are the hyperparameters you need to tune for better training result
```python
actor_lr=0.00005
critic_lr=0.002
```

You can define where to stop the training by adjust the termination condition in: 

```python
if (np.mean(return_list[-10:]) >= -20 or (not break_program) )and i>10:
        break
```
Press enter on your keyboard will terminate the training process too

Once the training is done, the trained model will be saved through:
```python
torch.save(agent.target_actor.state_dict(), 'RLPolicy2.pth') # You can change the name "RLPolicy2.pth"
```

In TestRLlqr.py
```python
rl.load_state_dict(torch.load('RLPolicy2.pth')) "replace with your trained model name"
```

and run to justify your trained model
