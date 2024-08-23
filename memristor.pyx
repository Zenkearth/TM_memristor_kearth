# cython: language_level=3

cdef class Memristor:
    def __init__(self, int ta_state, float init_memristor_state, int number_of_states,
                 float alpha_off, float alpha_on, float v_off, float v_on,
                 float r_off, float r_on, float k_off, float k_on, float d,
                 int clause, int feature, int negated):
        self.ta_state = ta_state
        self.init_memristor_state = init_memristor_state
        self.number_of_states = number_of_states
        self.alpha_off = alpha_off
        self.alpha_on = alpha_on
        self.v_off = v_off
        self.v_on = v_on
        self.r_off = r_off
        self.r_on = r_on
        self.k_off = k_off
        self.k_on = k_on
        self.d = d
        self.dx = 0

        self.mr_state = (ta_state/self.number_of_states)*self.init_memristor_state
        self.x = self.mr_state * self.d

    def tune(self, float voltage, float dt):
        if voltage > self.v_off:
            self.dx = self.k_off * (((voltage/self.v_off) - 1) ** self.alpha_off) * dt
        elif voltage < self.v_on:
            self.dx = self.k_on * (((voltage/self.v_on) - 1) ** self.alpha_on) * dt

        self.x += self.dx

        if self.x > self.d:
            self.x = self.d

        if self.x < 0:
            self.x = 0

        self.mr_state = self.x/self.d
        # print(self.mr_state, self.x, self.dx)

    def get_mr_state(self):
        return self.mr_state

    def get_mr_xdx(self):
        return int(self.mr_state*self.number_of_states/self.init_memristor_state), self.mr_state, self.dx, self.x

    def get_ta_state(self):
        return int(self.mr_state*self.number_of_states/self.init_memristor_state)