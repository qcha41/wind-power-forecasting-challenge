import utilities

rm = utilities.RunManager('test')
#rm.models[2].process()
rm.process_all_wf()

# for window_size in [6,12,24,48,96]:
#     rm = utilities.RunManager('test')
#     rm.models[2].window_size = window_size
#     rm.models[2].initialize()
#     rm.models[2].process()
#     print(window_size, rm.models[2].scores)

