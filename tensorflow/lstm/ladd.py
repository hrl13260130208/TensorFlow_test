import  numpy



def sigmoid(x):
    return 1./(1+numpy.exp(x))

def sigmoid_output_dervitive(output):
    return output*(1-output)

def tanh_derivative(values):
    return 1.-values**2


def random_arr(a,b,*args):
    numpy.random.seed(0)
    return numpy.random.rand(*args)*(b-a)+a

class LstmParam:
    def __init__(self,mem_cell_ct,x_dim):
        self.mem_cell_ct=mem_cell_ct
        self.x_dim=x_dim
        concat_len=x_dim+mem_cell_ct

        #weight matrices
        self.wg=random_arr(-0.1,0.1,mem_cell_ct,concat_len)
        self.wi=random_arr(-0.1,0.1,mem_cell_ct,concat_len)
        self.wf=random_arr(-0.1,0.1,mem_cell_ct,concat_len)
        self.wo=random_arr(-0.1,0.1,mem_cell_ct,concat_len)

        #bias terms
        self.bg=random_arr(-0.1,0.1,mem_cell_ct)
        self.bi=random_arr(-0.1,0.1,mem_cell_ct)
        self.bf=random_arr(-0.1,0.1,mem_cell_ct)
        self.bo=random_arr(-0.1,0.1,mem_cell_ct)

        #diffs
        self.wg_diff=numpy.zeros((mem_cell_ct,concat_len))
        self.wi_diff=numpy.zeros((mem_cell_ct,concat_len))
        self.wf_diff=numpy.zeros((mem_cell_ct,concat_len))
        self.wo_diff=numpy.zeros((mem_cell_ct,concat_len))

        self.bg_diff=numpy.zeros(mem_cell_ct)
        self.bi_diff=numpy.zeros(mem_cell_ct)
        self.bf_diff=numpy.zeros(mem_cell_ct)
        self.bo_diff=numpy.zeros(mem_cell_ct)

    def apply_diff(self,lr=1):
        self.wg -=lr*self.wg_diff
        self.wi -=lr*self.wi_diff
        self.wf -=lr*self.wf_diff
        self.wo -=lr*self.wo_diff

        self.bg -=lr*self.bg_diff
        self.bi -=lr*self.bi_diff
        self.bf -=lr*self.bf_diff
        self.bo -=lr*self.bo_diff

        #reset diffs to zero
        self.wg_diff=numpy.zeros_like(self.wg)
        self.wi_diff=numpy.zeros_like(self.wi)
        self.wf_diff=numpy.zeros_like(self.wf)
        self.wo_diff=numpy.zeros_like(self.wo)

        self.bg_diff=numpy.zeros_like(self.bg)
        self.bi_diff=numpy.zeros_like(self.bi)
        self.bf_diff=numpy.zeros_like(self.bf)
        self.bo_diff=numpy.zeros_like(self.bo)

class LstmState:
    def __init__(self,mem_cell_ct,x_dim):
        self.g = numpy.zeros(mem_cell_ct)
        self.i = numpy.zeros(mem_cell_ct)
        self.f= numpy.zeros(mem_cell_ct)
        self.o= numpy.zeros(mem_cell_ct)
        self.s= numpy.zeros(mem_cell_ct)
        self.h= numpy.zeros(mem_cell_ct)

        self.bottom_diff_h=numpy.zeros_like(self.h)
        self.bottom_diff_s=numpy.zeros_like(self.s)
        self.bottom_diff_x=numpy.zeros(x_dim)

class LstmNode:
    def __init__(self,lstm_param,lstm_state):

        self.state=lstm_state
        self.param=lstm_param

        self.x=None
        self.xc=None

    def bottom_data_is(self,x,s_prev=None,h_prev=None):

        if s_prev is None: s_prev = numpy.zeros_like(self.state.s)
        if h_prev is None: h_prev = numpy.zeros_like(self.state.h)

        self.s_prev = s_prev
        self.h_prev = h_prev

        xc= numpy.hstack((x,h_prev))
        self.state.g = numpy.tanh(numpy.dot(self.param.wg,xc)+self.param.bg)
        self.state.i = sigmoid(numpy.dot(self.param.wi,xc)+self.param.bi)
        self.state.f = sigmoid(numpy.dot(self.param.wf,xc)+self.param.bf)
        self.state.o = sigmoid(numpy.dot(self.param.wo,xc)+self.param.bo)
        self.state.s = self.state.g * self.state.i + s_prev * self.state.f
        self.state.h = self.state.s * self.state.o

        self.x = x
        self.xc=xc

    def top_diff_is(self,top_diff_h,top_diff_s):

        ds =self.state.o * top_diff_h+top_diff_s
        do =self.state.s * top_diff_h
        di =self.state.g * ds
        dg =self.state.i * ds
        df =self.s_prev * ds

        di_input =(1. - self.state.i) * self.state.i *di
        df_input =(1. - self.state.f) * self.state.f *df
        do_input =(1. - self.state.o) * self.state.o *do
        dg_input =(1. - self.state.g ** 2 ) *dg

        self.param.wi_diff += numpy.outer(di_input,self.xc)
        self.param.wf_diff += numpy.outer(df_input,self.xc)
        self.param.wo_diff += numpy.outer(do_input,self.xc)
        self.param.wg_diff += numpy.outer(dg_input,self.xc)

        self.param.bi_diff +=di_input
        self.param.bf_diff +=df_input
        self.param.bo_diff +=do_input
        self.param.bg_diff +=dg_input

        dxc = numpy.zeros_like(self.xc)
        dxc += numpy.dot(self.param.wi.T,di_input)
        dxc += numpy.dot(self.param.wf.T,df_input)
        dxc += numpy.dot(self.param.wo.T,do_input)
        dxc += numpy.dot(self.param.wg.T,dg_input)

        self.state.bottom_diff_s = ds * self.state.f
        self.state.bottom_diff_x = dxc[:self.param.x_dim]
        self.state.bottom_diff_h = dxc[self.param.x_dim:]


class LstmNetwork():
    def __init__(self,lstm_param):
        self.lstm_param=lstm_param
        self.lstm_node_list=[]
        #input sequence
        self.x_list=[]

    def y_list_is(self,y_list,loss_layer):

        assert  len(y_list) == len(self.x_list)
        idx = len(self.x_list) - 1

        loss= loss_layer.loss(self.lstm_node_list[idx].state.h,y_list[idx])
        diff_h = loss_layer.bottom_diff(self.lstm_node_list[idx].state.h,y_list[idx])

        diff_s=numpy.zeros(self.lstm_param.mem_cell_ct)
        self.lstm_node_list[idx].top_diff_is(diff_h,diff_s)
        idx -= 1

        while idx>= 0:
            loss += loss_layer.loss(self.lstm_node_list[idx].state.h,y_list[idx])
            diff_h=loss_layer.bottom_diff(self.lstm_node_list[idx].state.h,y_list[idx])
            diff_h += self.lstm_node_list[idx + 1].state.bottom_diff_h
            diff_s = self.lstm_node_list[idx + 1].state.bottom_diff_s
            self.lstm_node_list[idx].top_diff_is(diff_h,diff_s)
            idx -= 1

        return loss

    def x_list_clear(self):
        self.x_list=[]

    def x_list_add(self,x):
        self.x_list.append(x)
        if len(self.x_list)>len(self.lstm_node_list):
            lstm_state = LstmState(self.lstm_param.mem_cell_ct,self.lstm_param.x_dim)
            self.lstm_node_list.append(LstmNode(self.lstm_param,lstm_state))

        idx = len(self.x_list) - 1
        if idx == 0:
            self.lstm_node_list[idx].bottom_data_is(x)
        else:
            s_prev = self.lstm_node_list[idx - 1].state.s
            h_prev = self.lstm_node_list[idx - 1].state.h
            self.lstm_node_list[idx].bottom_data_is(x,s_prev,h_prev)

class ToyLossLayer:
    @classmethod
    def loss(self,pred,label):
        return (pred[0] - label)**2

    @classmethod
    def bottom_diff(self,pred,label):
        diff = numpy.zeros_like(pred)
        diff[0] = 2 * (pred[0] - label)
        return  diff



def example_0():
    numpy.random.seed(0)
    mem_cell_ct = 100
    x_dim = 50
    concat_len = x_dim + mem_cell_ct
    lstm_param = LstmParam(mem_cell_ct,x_dim)
    lstm_net = LstmNetwork(lstm_param)
    y_list=[-0.5,0.2,0.1,-0.5]
    input_val_arr = [numpy.random.random(x_dim) for  _ in y_list]

    for cur_iter in range(100):
        print("cur iter ", cur_iter)
        for ind in range(len(y_list)):
            lstm_net.x_list_add(input_val_arr[ind])
            print("y_pred[%d] : %f" % (ind, lstm_net.lstm_node_list[ind].state.h[0]))

        loss = lstm_net.y_list_is(y_list,ToyLossLayer)
        print("loss: ",loss)
        lstm_param.apply_diff(lr=0.1)
        lstm_net.x_list_clear()

class Primes:
    def __init__(self):
        self.primes = list()
        for i in range(2, 100):
            is_prime = True
            for j in range(2, i-1):
                if i % j == 0:
                    is_prime = False
            if is_prime:
                self.primes.append(i)
        self.primes_count = len(self.primes)
    def get_sample(self, x_dim, y_dim, index):
        result = numpy.zeros((x_dim+y_dim))
        for i in range(index, index + x_dim + y_dim):
            result[i-index] = self.primes[i%self.primes_count]/100.0
        return result


def example_1():
    mem_cell_ct = 100
    x_dim = 50
    concat_len = x_dim + mem_cell_ct
    lstm_param = LstmParam(mem_cell_ct, x_dim)
    lstm_net = LstmNetwork(lstm_param)

    primes = Primes()
    x_list = []
    y_list = []
    for i in range(0, 10):
        sample = primes.get_sample(x_dim, 1, i)
        x = sample[0:x_dim]
        y = sample[x_dim:x_dim+1].tolist()[0]
        x_list.append(x)
        y_list.append(y)

    for cur_iter in range(10000):
        if cur_iter % 1000 == 0:
            print ("y_list=", y_list)
        for ind in range(len(y_list)):
            lstm_net.x_list_add(x_list[ind])
            if cur_iter % 1000 == 0:
                print( "y_pred[%d] : %f" % (ind, lstm_net.lstm_node_list[ind].state.h[0]))

        loss = lstm_net.y_list_is(y_list, ToyLossLayer)
        if cur_iter % 1000 == 0:
            print ("loss: ", loss)
        lstm_param.apply_diff(lr=0.001)
        lstm_net.x_list_clear()

if __name__ == '__main__':
    example_1()














