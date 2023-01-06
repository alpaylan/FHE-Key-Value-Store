import concrete.numpy as cnp
import numpy as np
import time

NUMBER_OF_ENTRIES = 5

KEY_SIZE = 4
VALUE_SIZE = 4
CHUNK_SIZE = 4

NUMBER_OF_KEY_CHUNKS = KEY_SIZE//CHUNK_SIZE
NUMBER_OF_VALUE_CHUNKS = VALUE_SIZE//CHUNK_SIZE


FLAG = 0
KEY = slice(1, 1 + NUMBER_OF_KEY_CHUNKS)
VALUE = slice(1 + NUMBER_OF_KEY_CHUNKS, None)

STATE_SHAPE = (NUMBER_OF_ENTRIES, 1 + 1 + 1)


import concrete.numpy as cnp
import numpy as np

configuration = cnp.Configuration(
    enable_unsafe_features=True,
    use_insecure_key_cache=True,
    insecure_key_cache_location=".keys",
)

lut_zero = cnp.LookupTable([1, 0, 0, 0])

lut2 = cnp.LookupTable(
    [0 for _ in range(16)] + [i for i in range(16)]
)

def _insert_impl(state, key, value):
    flags = state[:, FLAG]
    
    selection = cnp.zeros(NUMBER_OF_ENTRIES)

    found = cnp.zero()
    for i in range(NUMBER_OF_ENTRIES):
        selection_i = lut_zero[(found * 2) + flags[i]]

        selection[i] = selection_i
        found += selection_i

    diff = cnp.zeros(STATE_SHAPE)
    diff[:, FLAG] = selection

    selection = selection.reshape((-1, 1))
    key_diff = lut2[selection * (2 ** CHUNK_SIZE) + key]
    val_diff = lut2[selection * (2 ** CHUNK_SIZE) + value]
    
    diff[:, KEY] = key_diff
    diff[:, VALUE] = val_diff
    return state + diff

# state = np.array([
#     [1, 2, 1],
#     [0, 0, 0],
#     [0, 0, 0],
#     [0, 0, 0],
#     [0, 0, 0],
# ])

# key = np.array([3])
# value = np.array([4])

# print(_insert_impl(state, key, value))


def _replace_impl(state, key, value):
    keys = state[:, KEY]
    values = state[:, VALUE]

    selection = cnp.zeros(NUMBER_OF_ENTRIES)

    for i in range(NUMBER_OF_ENTRIES):
        selection[i] = (keys[i] - key == 0)

    inverse_selection = 1 - selection

    selection = selection.reshape((-1, 1))
    inverse_selection = inverse_selection.reshape((-1, 1))
    

    new_value = lut2[selection * (2 ** CHUNK_SIZE) + value] + lut2[inverse_selection * (2 ** CHUNK_SIZE) + values]

    new_state = cnp.zeros(STATE_SHAPE)
    new_state[:, FLAG] = state[:, FLAG]
    new_state[:, KEY] = state[:, KEY]
    new_state[:, VALUE] = new_value

    return new_state


def _query_impl(state, key):
    keys = state[:, KEY]
    values = state[:, VALUE]


    selection = cnp.zeros(NUMBER_OF_ENTRIES)

    for i in range(NUMBER_OF_ENTRIES):
        selection[i] = (keys[i] - key == 0)

    found = np.sum(selection)
    selection = selection.reshape((-1, 1))

    val_diff = lut2[selection * (2 ** CHUNK_SIZE) + values]

    result = np.sum(val_diff)

    return cnp.array([found, result])

class KeyValueDatabase:

    number_of_entries: int
    
    _insert_circuit: cnp.Circuit
    _replace_circuit: cnp.Circuit
    _query_circuit: cnp.Circuit
    _state: np.ndarray


    def __init__(self):
        inputset_binary = [
            (
                np.ones(STATE_SHAPE, dtype=np.int64), # state
                np.ones(NUMBER_OF_KEY_CHUNKS, dtype=np.int64) * (2**CHUNK_SIZE - 1), # key
            ),
            (
                np.zeros(STATE_SHAPE, dtype=np.int64), # state
                np.ones(NUMBER_OF_KEY_CHUNKS, dtype=np.int64) * (2**CHUNK_SIZE - 1), # key
            ),
        ]
        
        inputset_ternary = [
            (
                np.ones(STATE_SHAPE, dtype=np.int64), # state
                np.ones(NUMBER_OF_KEY_CHUNKS, dtype=np.int64) * (2**CHUNK_SIZE - 1), # key
                np.ones(NUMBER_OF_VALUE_CHUNKS, dtype=np.int64) * (2**CHUNK_SIZE - 1), # value
            ),
            (
                np.zeros(STATE_SHAPE, dtype=np.int64), # state
                np.ones(NUMBER_OF_KEY_CHUNKS, dtype=np.int64) * (2**CHUNK_SIZE - 1), # key
                np.ones(NUMBER_OF_VALUE_CHUNKS, dtype=np.int64) * (2**CHUNK_SIZE - 1), # value
            )
        ]
        configuration = cnp.Configuration(
            enable_unsafe_features=True,
            use_insecure_key_cache=True,
            insecure_key_cache_location=".keys",
            verbose=True,
            virtual=True,
        )

        self._state = np.zeros(STATE_SHAPE, dtype=np.int64)
        
        insert_compiler = cnp.Compiler(_insert_impl, {"state": "encrypted", "key": "encrypted", "value": "encrypted"})
        self._insert_circuit = insert_compiler.compile(inputset_ternary, configuration)

        replace_compiler = cnp.Compiler(_replace_impl, {"state": "encrypted", "key": "encrypted", "value": "encrypted"})
        self._replace_circuit = replace_compiler.compile(inputset_ternary, configuration)

        query_compiler = cnp.Compiler(_query_impl, {"state": "encrypted", "key": "encrypted"})
        self._query_circuit = query_compiler.compile(inputset_binary, configuration)


    def insert(self, key, value):
        start = time.time()
        self._state = self._insert_circuit.encrypt_run_decrypt(self._state, key, value)
        end = time.time()
        print(f"(took {end - start:.3f} seconds)")
        
        print("State:", self._state)

    def replace(self, key, value):
        start = time.time()
        self._state = self._replace_circuit.encrypt_run_decrypt(self._state, key, value)
        end = time.time()
        print(f"(took {end - start:.3f} seconds)")
        
        print("State:", self._state)

    def query(self, key):
        start = time.time()
        result = self._query_circuit.encrypt_run_decrypt(self._state, key)
        end = time.time()
        print(f"(took {end - start:.3f} seconds)")

        print(result)

        if result[0] == 0:
            return None

        return result[1]


db = KeyValueDatabase()

# Test: Insert/Query
db.insert([3], [4])
assert db.query([3]) == 4

db.replace([3], [1])
assert db.query([3]) == 1

# Test: Insert/Query
db.insert([2], [3])
assert db.query([2]) == 3

# Test: Query Not Found
assert db.query([4]) == None

# Test: Replace/Query
db.replace([3], [5])
assert db.query([3]) == 5
