from core.praser import init_obj

phase: str
init_type: str
model_opt: dict


def create_model(model_opt):
    """ create_model """
    model = init_obj(model_opt, default_file_name='models.model', init_type='Model')

    return model


def define_network(network_opt):
    """ define network with weights initialization """
    net = init_obj(network_opt, default_file_name='models.network', init_type='Network')

    if phase == 'train':
        print(f'Network [{net.__class__.__name__}] weights initialize using [{init_type}] method.')
        net.init_weights()
    return net

loss_opt: dict
def define_loss(loss_opt):
    return init_obj(loss_opt, default_file_name='models.loss', init_type='Loss')

def define_metric(metric_opt):
    return init_obj(metric_opt, default_file_name='models.metric', init_type='Metric')

