import datetime
import numpy as np
import heapq
import queue 
import threading
import json

class myheapobj():
    def __init__(self, value, slice, index):
        self._value = value
        self._slice = slice
        self._index = index
    
    def __lt__(self, other):
        try:
            return (int)(self._value[0]) < (int)(other._value[0])
        except AttributeError:
            return NotImplemented

def merge(slices):
    heap = []
    for slice in slices:
        heapobj = myheapobj(slice[0], slice, 0)
        heap.append(heapobj)
    
    heapq.heapify(heap)

    ids = []  #list of ids
    checkins = []  #list of checkins
    last_id = None
    duplicates = 0
    while(len(heap) > 0):
        x = heapq.heappop(heap)
        if last_id == None or last_id != x._value[0]:
            ids.append(x._value[0])
            #if len(x._value) == 4:
            #    checkins.append(str(x._value[1]) + ' ' + str(x._value[2]) + ' ' + str(x._value[3]))
            if len(x._value) > 2:
                checkins.append(str(x._value[1:].tolist()).replace(',', '').replace('[', '').replace(']', ''))
            else:
                checkins.append(x._value[1])
            last_id = x._value[0]
        else:
            duplicates += 1
        if x._index < len(x._slice) - 1:
            heapobj = myheapobj(x._slice[x._index + 1], x._slice, x._index + 1)
            heapq.heappush(heap, heapobj)
       
    print('records', len(ids))
    print('duplicates', duplicates)
      
    return np.column_stack((ids, checkins))

def merge2(slices, submit = False, n_topx = 3):
    heap = []
    for slice in slices:
        heapobj = myheapobj(slice[0], slice, 0)
        heap.append(heapobj)
    
    heapq.heapify(heap)

    ids = []  #list of ids
    places = []  #list dictionaries { place : score }
    last_id = None
    last_dict = {}

    duplicates = 0

    while(len(heap) > 0):
        x = heapq.heappop(heap)
        id = x._value[0]
        dict = json.loads(x._value[1])
        dict2 = {}
        for key, value in dict.items():
            dict2[int(key)]=float(value)
        dict = dict2

        if id == last_id:
            duplicates += 1
            for key, value in dict.items():
                if last_dict.get(key) != None:
                    last_dict[key] += value
                else:
                    last_dict[key] = value

        else:
            if last_id != None:
                 ids.append(last_id)
                 if submit == False:
                     places.append(json.dumps(last_dict))
                 else:
                     top3 = sorted(last_dict, key=last_dict.__getitem__, reverse=True)[:n_topx]
                     #places.append(str(top3[0]) + ' ' + str(top3[1]) + ' ' + str(top3[2]))
                     places.append(str(top3).replace(',', '').replace('[', '').replace(']', ''))

            last_id = id
            last_dict = dict
                
        if x._index < len(x._slice) - 1:
            heapobj = myheapobj(x._slice[x._index + 1], x._slice, x._index + 1)
            heapq.heappush(heap, heapobj)
    
    #push last id       
    ids.append(last_id)
    if submit == False:
        places.append(json.dumps(last_dict))
    else:
        top3 = sorted(last_dict, key=last_dict.__getitem__, reverse=True)[:n_topx]
        #places.append(str(top3[0]) + ' ' + str(top3[1]) + ' ' + str(top3[2]))
        places.append(str(top3).replace(',', '').replace('[', '').replace(']', ''))

    print('records', 'duplicates', len(ids), duplicates)
      
    return np.column_stack((ids, places))

class myThread2 (threading.Thread):
    def __init__(self, tasks, function, results):
        threading.Thread.__init__(self)
        self._tasks = tasks
        self._function = function
        self._results = results
    
    def run(self):
        while not self._tasks.empty():
            task = None
            try:
                task = self._tasks.get(block=False)
            except queue.Empty:
                print('Ignore empty exception')

            if task != None:
                result = self._function(task)
                self._results.append(result)


def run_tasks(tasks, function, n_threads = 10):
    n_tasks = tasks.qsize()
    print('tasks start', n_tasks, datetime.datetime.now())
    start_time = datetime.datetime.now()
        
    threads = []
    results = []

    for i in range (n_threads):
        thread = myThread2(tasks, function, results)
        thread.start()
        threads.append(thread)

    for t in threads:
        t.join()

    print('tasks done', n_tasks, (datetime.datetime.now() - start_time).seconds//60, datetime.datetime.now())
    return results
