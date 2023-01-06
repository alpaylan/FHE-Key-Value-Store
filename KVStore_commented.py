import time

import concrete.numpy as cnp
import numpy as np


# The number of entries in the database
NUMBER_OF_ENTRIES = 5
# The number of bits in each chunk
CHUNK_SIZE = 4

# The number of bits in the key and value
KEY_SIZE = 32
VALUE_SIZE = 32

# Key and Value size must be a multiple of chunk size
assert KEY_SIZE % CHUNK_SIZE == 0
assert VALUE_SIZE % CHUNK_SIZE == 0

# Required number of chunks to store keys and values
NUMBER_OF_KEY_CHUNKS = KEY_SIZE // CHUNK_SIZE
NUMBER_OF_VALUE_CHUNKS = VALUE_SIZE // CHUNK_SIZE

# The shape of the state as a tensor
# Shape:
# | Flag Size | Key Size | Value Size |
# | 1         | 32/4 = 8 | 32/4 = 8   |
STATE_SHAPE = (NUMBER_OF_ENTRIES, 1 + NUMBER_OF_KEY_CHUNKS + NUMBER_OF_VALUE_CHUNKS)

# Indexers for each part of the state
FLAG = 0
KEY = slice(1, 1 + NUMBER_OF_KEY_CHUNKS)
VALUE = slice(1 + NUMBER_OF_KEY_CHUNKS, None)



## Encode/Decode Functions for conversions from Int <-> Numpy Array

# Encode a number into a numpy array
# The number is encoded in binary and then split into chunks
# Each chunk is then converted to an integer
# The integers are then stored in a numpy array
# Example:
#   encode(10, 4) -> [1, 0, 1, 0]
#   encode(11, 5) -> [0, 1, 0, 1, 1]
def encode(number: int, width: int) -> np.array:
    binary_repr = np.binary_repr(number, width=width)
    blocks = [binary_repr[i:i+CHUNK_SIZE] for i in range(0, len(binary_repr), CHUNK_SIZE)]
    return np.array([int(block, 2) for block in blocks])

# Encode a number with the key size
def encode_key(number: int) -> np.array:
    return encode(number, width=KEY_SIZE)
# Encode a number with the value size
def encode_value(number: int) -> np.array:
    return encode(number, width=VALUE_SIZE)

# Decode a numpy array into a number
# The numpy array is converted to integer
# Example:
#   decode([1, 0, 1, 0]) -> 10
#   decode([0, 1, 0, 1, 1]) -> 11
def decode(encoded_number: np.array) -> int:
    result = 0
    for i in range(len(encoded_number)):
        result += 2**(CHUNK_SIZE*i) * encoded_number[(len(encoded_number) - i) - 1]
    return result


# The lookup table for the following function
# def keep_selected(value, selected):
#   if selected:
#       return value
#   else:
#       return 0
# Selected is pushed as the first bit of the value, 
# and the lookup table is used to extract the value
# Example:
#   keep_selected(i=0..15, 1) -> 0
#   keep_selected(i=16..31, 0) -> i
keep_selected_lut = cnp.LookupTable([0 for _ in range(16)] + [i for i in range(16)])


## Circuit Implementation Functions
# These functions are used to implement the kv store circuits.
# Three circuits are implemented:
#   insert: Inserts a key value pair into the database
#   replace: Replaces the value of a key in the database
#   query: Queries the database for a key and returns the value

# Insert a key value pair into the database
# - state: The state of the database
# - key: The key to insert
# - value: The value to insert
# Returns the updated state
def _insert_impl(state, key, value):
    # Get the used bit from the state
    # This bit is used to determine if an entry is used or not
    flags = state[:, FLAG]

    # Create a selection array
    # This array is used to select the first unused entry
    selection = cnp.zeros(NUMBER_OF_ENTRIES)

    # The found bit is used to determine if an unused entry has been found
    found = cnp.zero()
    for i in range(NUMBER_OF_ENTRIES):
        # The packed flag and found bit are used to determine if the entry is unused
        # | Flag | Found |
        # | 0    | 0     | -> Unused, select
        # | 0    | 1     | -> Unused, skip
        # | 1    | 0     | -> Used, skip
        # | 1    | 1     | -> Used, skip
        packed_flag_and_found = (found * 2) + flags[i]
        # Use the packed flag and found bit to determine if the entry is unused
        is_selected = (packed_flag_and_found == 0)

        # Update the selection array
        selection[i] = is_selected
        # Update the found bit, so all entries will be 
        # skipped after the first unused entry is found
        found += is_selected

    # Create a state update array
    state_update = cnp.zeros(STATE_SHAPE)
    # Update the state update array with the selection array
    state_update[:, FLAG] = selection

    # Reshape the selection array to be able to use it as an index
    selection = selection.reshape((-1, 1))

    # Create a packed selection and key array
    # This array is used to update the key of the selected entry
    packed_selection_and_key = (selection * (2 ** CHUNK_SIZE)) + key
    key_update = keep_selected_lut[packed_selection_and_key]

    # Create a packed selection and value array
    # This array is used to update the value of the selected entry
    packed_selection_and_value = selection * (2 ** CHUNK_SIZE) + value
    value_update = keep_selected_lut[packed_selection_and_value]

    # Update the state update array with the key and value update arrays
    state_update[:, KEY] = key_update
    state_update[:, VALUE] = value_update

    # Update the state with the state update array
    new_state = state + state_update
    return new_state

# Replace the value of a key in the database
#   If the key is not in the database, nothing happens
#   If the key is in the database, the value is replaced
# - state: The state of the database
# - key: The key to replace
# - value: The value to replace
# Returns the updated state
def _replace_impl(state, key, value):
    # Get the keys and values from the state
    keys = state[:, KEY]
    values = state[:, VALUE]

    # Create a selection array
    # This array is used to select the entry with the given key
    # The selection array is created by comparing the keys in the state
    # with the given key, and only setting the entry to 1 if the keys are equal
    # Example:
    #   keys = [[1, 0, 1, 0], [0, 1, 0, 1, 1]]
    #   key = [1, 0, 1, 0]
    #   selection = [1, 0]
    selection = (np.sum((keys - key) == 0, axis=1) == NUMBER_OF_KEY_CHUNKS).reshape((-1, 1))

    # Create a packed selection and value array
    # This array is used to update the value of the selected entry
    packed_selection_and_value = selection * (2 ** CHUNK_SIZE) + value
    set_value = keep_selected_lut[packed_selection_and_value]

    # Create an inverse selection array
    # This array is used to pick entries that are not selected
    # Example:
    #   selection = [1, 0, 0]
    #   inverse_selection = [0, 1, 1]
    inverse_selection = 1 - selection

    # Create a packed inverse selection and value array
    # This array is used to keep the value of the entries that are not selected
    packed_inverse_selection_and_values = inverse_selection * (2 ** CHUNK_SIZE) + values
    kept_values = keep_selected_lut[packed_inverse_selection_and_values]

    # Update the values of the state with the new values
    new_values = kept_values + set_value
    state[:, VALUE] = new_values

    return state

# Query the database for a key and return the value
# - state: The state of the database
# - key: The key to query
# Returns an array with the following format:
#   [found, value]
#   found: 1 if the key was found, 0 otherwise
#   value: The value of the key if the key was found, 0 otherwise
def _query_impl(state, key):
    # Get the keys and values from the state
    keys = state[:, KEY]
    values = state[:, VALUE]

    # Create a selection array
    # This array is used to select the entry with the given key
    # The selection array is created by comparing the keys in the state
    # with the given key, and only setting the entry to 1 if the keys are equal
    # Example:
    #   keys = [[1, 0, 1, 0], [0, 1, 0, 1, 1]]
    #   key = [1, 0, 1, 0]
    #   selection = [1, 0]
    selection = (np.sum((keys - key) == 0, axis=1) == NUMBER_OF_KEY_CHUNKS).reshape((-1, 1))

    # Create a found bit
    # This bit is used to determine if the key was found
    # The found bit is set to 1 if the key was found, and 0 otherwise
    found = np.sum(selection)

    # Create a packed selection and value array
    # This array is used to get the value of the selected entry
    packed_selection_and_values = selection * (2 ** CHUNK_SIZE) + values
    value_selection = keep_selected_lut[packed_selection_and_values]

    # Sum the value selection array to get the value
    value = np.sum(value_selection, axis=0)

    # Return the found bit and the value
    return cnp.array([found, *value])


class KeyValueDatabase:
    """
    A key-value database that uses fully homomorphic encryption circuits to store the data.
    """

    # The state of the database, it holds all the 
    # keys and values as a table of entries
    _state: np.ndarray

    # The circuits used to implement the database
    _insert_circuit: cnp.Circuit
    _replace_circuit: cnp.Circuit
    _query_circuit: cnp.Circuit

    def __init__(self):
        # Initialize the state to all zeros
        self._state = np.zeros(STATE_SHAPE, dtype=np.int64)

        ## Input sets for initialization of the circuits
        # The input sets are used to initialize the circuits with the correct parameters

        # The input set for the query circuit
        inputset_binary = [
            (
                np.zeros(STATE_SHAPE, dtype=np.int64), # state
                np.ones(NUMBER_OF_KEY_CHUNKS, dtype=np.int64) * (2**CHUNK_SIZE - 1), # key
            )
        ]
        # The input set for the insert and replace circuits
        inputset_ternary = [
            (
                np.zeros(STATE_SHAPE, dtype=np.int64), # state
                np.ones(NUMBER_OF_KEY_CHUNKS, dtype=np.int64) * (2**CHUNK_SIZE - 1), # key
                np.ones(NUMBER_OF_VALUE_CHUNKS, dtype=np.int64) * (2**CHUNK_SIZE - 1), # value
            )
        ]

        ## Circuit compilation

        # Create a configuration for the compiler
        configuration = cnp.Configuration(
            enable_unsafe_features=True,
            use_insecure_key_cache=True,
            insecure_key_cache_location=".keys",
            # virtual=True,
        )

        # Create the compilers for the circuits
        # Each compiler is provided with
        # - The implementation of the circuit
        # - The inputs and their corresponding types of the circuit
        #  - "encrypted": The input is encrypted
        #  - "plain": The input is not encrypted
        insert_compiler = cnp.Compiler(
            _insert_impl,
            {"state": "encrypted", "key": "encrypted", "value": "encrypted"}
        )
        replace_compiler = cnp.Compiler(
            _replace_impl,
            {"state": "encrypted", "key": "encrypted", "value": "encrypted"}
        )
        query_compiler = cnp.Compiler(
            _query_impl,
            {"state": "encrypted", "key": "encrypted"}
        )


        ## Compile the circuits
        # The circuits are compiled with the input set and the configuration

        print()

        print("Compiling insertion circuit...")
        start = time.time()
        self._insert_circuit = insert_compiler.compile(inputset_ternary, configuration)
        end = time.time()
        print(f"(took {end - start:.3f} seconds)")

        print()

        print("Compiling replacement circuit...")
        start = time.time()
        self._replace_circuit = replace_compiler.compile(inputset_ternary, configuration)
        end = time.time()
        print(f"(took {end - start:.3f} seconds)")

        print()

        print("Compiling query circuit...")
        start = time.time()
        self._query_circuit = query_compiler.compile(inputset_binary, configuration)
        end = time.time()
        print(f"(took {end - start:.3f} seconds)")

        print()

        ## Generate the keys for the circuits
        # The keys are seaparately generated for each circuit

        print("Generating insertion keys...")
        start = time.time()
        self._insert_circuit.keygen()
        end = time.time()
        print(f"(took {end - start:.3f} seconds)")

        print()

        print("Generating replacement keys...")
        start = time.time()
        self._replace_circuit.keygen()
        end = time.time()
        print(f"(took {end - start:.3f} seconds)")

        print()

        print("Generating query keys...")
        start = time.time()
        self._query_circuit.keygen()
        end = time.time()
        print(f"(took {end - start:.3f} seconds)")

    # The following methods are used to interact with the database
    # They are used to insert, replace and query the database
    # The methods are implemented by encrypting the inputs, running the circuit and decrypting the output

    # Insert a key-value pair into the database
    # - key: The key to insert
    # - value: The value to insert
    # The key and value are encoded before they are inserted
    # The state of the database is updated with the new key-value pair
    def insert(self, key, value):
        print()
        print(f"Inserting...")
        start = time.time()
        self._state = self._insert_circuit.encrypt_run_decrypt(
            self._state, encode_key(key), encode_value(value)
        )
        end = time.time()
        print(f"(took {end - start:.3f} seconds)")

    # Replace a key-value pair in the database
    # - key: The key to replace
    # - value: The new value to insert with the key
    # The key and value are encoded before they are inserted
    # The state of the database is updated with the new key-value pair
    def replace(self, key, value):
        print()
        print(f"Replacing...")
        start = time.time()
        self._state = self._replace_circuit.encrypt_run_decrypt(
            self._state, encode_key(key), encode_value(value)
        )
        end = time.time()
        print(f"(took {end - start:.3f} seconds)")

    # Query the database for a key
    # - key: The key to query
    # The key is encoded before it is queried
    # Returns the value associated with the key or None if the key is not found
    def query(self, key):
        print()
        print(f"Querying...")
        start = time.time()
        result = self._query_circuit.encrypt_run_decrypt(
            self._state, encode_key(key)
        )
        end = time.time()
        print(f"(took {end - start:.3f} seconds)")

        if result[0] == 0:
            return None

        return decode(result[1:])


## Test: Initialization
# Initialize the database
db = KeyValueDatabase()

## Test: Insert/Query
# Insert (key: 3, value: 4) into the database
db.insert(3, 4)
# Query the database for the key 3
# The value 4 should be returned
assert db.query(3) == 4

## Test: Replace/Query
# Replace the value of the key 3 with 1
db.replace(3, 1)
# Query the database for the key 3
# The value 1 should be returned
assert db.query(3) == 1

## Test: Insert/Query
# Insert (key: 25, value: 40) into the database
db.insert(25, 40)
# Query the database for the key 25
# The value 40 should be returned
assert db.query(25) == 40

## Test: Query Not Found
# Query the database for the key 4
# None should be returned
assert db.query(4) == None

## Test: Replace/Query
# Replace the value of the key 3 with 5
db.replace(3, 5)
# Query the database for the key 3
# The value 5 should be returned
assert db.query(3) == 5
