'''
Configurations for the simple CapsNet architecture.
'''

defaults = {
    'image_size': 64,
    'image_depth' : 1,
    'learning_rate': 1e-4,
    'decay_rate': 0.99,
    'primary_caps_vlength': 16,
    'digit_caps_vlength': 32,
    'epsilon': 1e-9,
    'lambda_': 0.5,
    'm_plus': 0.9,
    'm_minus': 0.1,
    'reg_scale': 0.005,
    'routing_iteration': 3,
    'num_classes':2
}
