from ctypes import *
from queue import Queue
from threading import Thread
import socket
import json
import os.path

libDir = "lib"
hashing_name = libDir + os.path.sep + "hashing.so"
hexutil_name = libDir + os.path.sep + "hexutil.so"
config_name = libDir + os.path.sep +  "config.so"
stringutil_name = libDir + os.path.sep + "stringutil.so"

hashing_path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + hashing_name
hexutil_path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + hexutil_name
config_path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + config_name
stringutil_path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + stringutil_name

# note: these files must first be built. see the make file
hashing = CDLL(hashing_path)
hexutil = CDLL(hexutil_path)
config = CDLL(config_path)
stringutil = CDLL(stringutil_path)

class BytesResponse(Structure):
    _fields_ = [
        ("r0", c_void_p),
        ("r1", c_int),
    ]

stringutil.CompactJSON.restype = BytesResponse
hexutil.DecodeString.restype = BytesResponse

ErrMethodAlreadyRegistered = Exception("method already registered")
ErrMethodNotExists = Exception("method does not exist")
ErrIncorrectNumberOfArgs = Exception("method requires two arguments")

class C3():
    def __init__(self, statefile):
        self.methods = {}
        self.state = {}
        self.q = Queue(maxsize=0)
        self.statefile = statefile

    def registerMethod(self, methodName, ifn):
        b = bytearray()
        b.extend(map(ord, methodName))
        arr = (c_byte * len(b))(*b)
        
        methodNameHash = c_char_p(hashing.HashToHexString(arr, len(arr))).value.decode('utf-8')
        if methodNameHash in self.methods:
            raise ErrMethodAlreadyExists

        def newMethod(*args):
            if len(args) != 2:
                raise ErrIncorrectNumberOfArgs

            key = args[0]
            res = hexutil.DecodeString(c_char_p(key.encode('utf-8')))
            ArrayType = c_ubyte*(c_int(res.r1).value)
            pa = cast(c_void_p(res.r0), POINTER(ArrayType))
            key = "".join(map(chr, pa.contents[:]))

            val = args[1]
            res = hexutil.DecodeString(c_char_p(val.encode('utf-8')))
            ArrayType = c_ubyte*(c_int(res.r1).value)
            pa = cast(c_void_p(res.r0), POINTER(ArrayType))
            val = "".join(map(chr, pa.contents[:]))

            try:
                res = ifn(key, val)
                print("[c3] result", res)
            except Exception as inst:
                print("[c3] invokation failed", inst)

        self.methods[methodNameHash] = newMethod
            
    def setInitialState(self):
        currState = ""

        with open(self.statefile, "r") as file:
            currState = file.read() 

        if len(currState) == 0:
            print("no current state")
            return

        b = bytearray()
        b.extend(map(ord, currState))
        arr = (c_byte * len(b))(*b)

        res = stringutil.CompactJSON(arr, len(arr))
        ArrayType = c_ubyte*(c_int(res.r1).value)
        pa = cast(c_void_p(res.r0), POINTER(ArrayType))

        self.state = json.loads("".join(map(chr, pa.contents[:])))
        print("initial state loaded")

    def process(self, payloadBytes):
        payload = json.loads(payloadBytes)

        if len(payload) <= 1:
            return

        # ifc format is [a, b, c]
        if isinstance(payload[0], str):
            self.invoke(payload[0], payload[1:])

        # ifc format is [[a, b, c], [a, b, c]]
        for ifc in payload:
            self.invoke(ifc[0], *ifc[1:])

    def invoke(self, methodName, *params):
        b = bytearray()
        b.extend(map(ord, methodName))
        arr = (c_byte * len(b))(*b)
        
        methodNameHash = c_char_p(hashing.HashToHexString(arr, len(arr))).value.decode('utf-8')
        if methodNameHash not in self.methods:
            raise ErrMethodNotExists

        fn = self.methods[methodNameHash]
        try:
            fn(*params)
        except Exception as inst:
            print("[c3] err invoking method", inst)
            return
    
    def listen(self):
        while True:
            self.process(self.q.get())
            q.task_done()

    def serve(self):
        host = c_char_p(config.ServerHost()).value
        port = c_int(config.ServerPort()).value

        server = Server(host, port, self.q)

        worker = Thread(target=server.run)
        # worker.setDaemon(True)
        worker.start()

def NewC3(stateFilePath = c_char_p(config.TempContainerStateFilePath()).value.decode('utf-8')):
    c3 = C3(stateFilePath)

    c3.setInitialState()

    worker = Thread(target=c3.listen)
    # worker.setDaemon(True)
    worker.start()

    return c3

class Server():
    def __init__(self, host, port, q):
        self.host = host
        self.port = port
        self.q = q

    def run(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((self.host, self.port))
        server.listen(1)  # max backlog of connections

        print("Listening on {0}:{1}".format(self.host, self.port))


        def handle_conn(conn):
            data = []
            while 1:
                tmp = conn.recv(1024)
                if not tmp: break
                data.append(tmp)

            self.q.put(''.join(data))
            conn.close()

        while True:
            client_sock, address = server.accept()
            print('Accepted connection from {0}:{1}'.format(address[0], address[1]))
            client_handler = threading.Thread(
                target=handle_conn,
                args=(client_sock,)  # note: comment required!
            )

            client_handler.start()
