import numpy as np

def iso_pointpillars(inp, gt_state, gt_class, get_cs, model, predict, monotone=True):
    "Adapted from https://github.com/matthewwicker/IterativeSalienceOcclusion"
    
    confidences = [1]
    removed = []
    removed_ind = []
    points_occluded = 0
    x = list(inp)
    y = gt_class
    
    # First you have to calculate an all pairs distance
    # and keep a matrix of points with the smallest change
    nearest_neighbors = []
    for i in range(len(x)):
        replace = x[0]
        del x[0]
        nearest_neighbors.append(closest_node(replace, x))
        x.append(replace)
        
    conf_i, cl = predict(x, model)
    if(cl != y):
        return 0,0,0,0
    
    # Calculate the critical set
    iterations = 0
    while(True):
        iterations += 1
        cs = get_cs(model, x)
        card_cs = len(cs)
        # Invert sort the critical set by the distance to the nearest neighbor
        cs = [c for _,c in sorted(zip(nearest_neighbors,cs))]
        cs = reversed(cs)
        
        # Manipulate and undo manipulations that increase the confidence 
        # of the network
        att = 0
        for i in cs:
            _replace = x[i]
            x[i] = [0,0,0] 
            conf, cl = predict(x,model)
            #if(monotone and iterations > 1800):
            #    montone = False
            if(monotone and conf <= conf_i):
                conf_i = conf
            elif(monotone):
                x[i] = _replace
                att += 1
                # sys.stdout.write("Point missed: %s \r"%(att))
                # sys.stdout.flush()
                if(att > card_cs-2):
                    print("Non-monotonic")
                    return 1024, -1, -1, -1
                continue
            # sys.stdout.write("Points occluded: %s conf: %s \r"%(points_occluded, conf))
            # sys.stdout.flush()
            if(cl != y):
                # lets refine the adversarial example:
                actually_removed = []
                actually_removed_ind = []
                for i in range(len(removed)):
                    x[removed_ind[i]] = removed[i]
                    conf, cl = predict(x,model)
                    if(cl == y):
                        x[removed_ind[i]] = [0,0,0]
                        actually_removed.append(removed[i])
                        actually_removed_ind.append(removed_ind[i])
                        points_occluded -= 1
                    #sys.stdout.write("Points occluded: %s conf: %s \r"%(points_occluded, conf))
                    #sys.stdout.flush()
                #print " "
                return len(actually_removed), x, actually_removed, actually_removed_ind                 
            # Without refinement
            #if(cl != y):
            #    return len(removed), x, removed, removed_ind 
            #if(conf >= confidences[-1] and iterations == 0):
            #    x[i] = _replace
            #    continue
            removed.append(_replace)
            removed_ind.append(i)
            points_occluded += 1
            confidences.append(conf)
        conf, cl = predict(x,model)
        if(points_occluded > 1024 and monotone == False):
            print("STARTING OVER")
            return -1, -1, -1, -1