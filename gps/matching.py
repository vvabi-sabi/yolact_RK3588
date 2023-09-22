from multiprocessing import Queue
import cv2
import numpy as np

from gps_utils import contour_map, mask2cnts, mad_cost, lstsq_affine_matrix, get_iou_cost, get_gps_data
from filters import create_kalman_filter, tri_shape_filter, get_point_ids_around_tri

class Matcher:

    AZIMUTH = -0 /180 *np.pi

    def __init__(self, map_path, gt_path, kalman=True) -> None:
        self.map_image = cv2.imread(map_path)
        t_masks = cv2.imread(gt_path)
        self.gt_mask = np.where(t_masks, 255, 0).astype(np.uint8)
        t_cnts, _ = cv2.findContours(self.gt_mask, 1, 2)
        t_cnts = [cnt for cnt in t_cnts if cv2.contourArea(cnt)>=16]
        self.tm = contour_map(t_cnts, 0)
        self.kalman = kalman
        self.filter = create_kalman_filter()
        #q_cnts = mask2cnts(q_mask)
        #self.qm = contour_map(q_cnts, self.AZIMUTH)
    
    def get_gps_data(self, frame, mask):
        gps_data = None
        im_h, im_w = frame.shape[:2]
        q_mask = cv2.resize(mask, frame.shape[-2::-1])
        q_cnts = mask2cnts(q_mask)
        q_mask = q_mask[:,:,0]

        if len(q_cnts) == 0:
            return frame, gps_data
        qm = contour_map(q_cnts, self.AZIMUTH)

        for qid, _ in enumerate(qm['simps']):
            loc_ok = False
            if -1 in qm['tri'].neighbors[qid]:
                continue

            # calculate correspondence metric between camera simplex and map simplices
            tids, reffmean, damean = tri_shape_filter(
                self.tm,
                qm['trads'][qid],
                qm['tangles'][qid]
            )

            # find points around query simplex and take object parameters for these points
            qnbr_ids = get_point_ids_around_tri(
                        qm,
                        qid,
                        level=6,
                        include_src_tri=True # ["use_src_tri"]
                    )
            qpts = qm['centers'][qnbr_ids]
            q_centers_pts = qpts - qm['tcenter'][qid]

            # calculate correspondence metrics for query and train points sets
            # only best correspondences from previous step used
            for tid in tids[:4]:
                tnbr_ids = get_point_ids_around_tri(
                        self.tm,
                        tid,
                        level=8,
                        include_src_tri=True
                    )
                tpts = self.tm['centers'][tnbr_ids]
                tareas = self.tm['areas'][tnbr_ids]

                proj = q_centers_pts * reffmean[tid]

                angle = -self.AZIMUTH + damean[tid]
                RotM = np.asarray(
                        [[np.cos(angle), np.sin(angle)],
                        [-np.sin(angle), np.cos(angle)]]
                    )

                proj = proj.dot(RotM.T) + self.tm['tcenter'][tid]

                norm_costs, rows, cols = mad_cost(proj, tpts, tareas)
                matched_point_ids = np.where(norm_costs<0.3)
                matched_qpts = qpts[rows[matched_point_ids]]
                matched_tpts = tpts[cols[matched_point_ids]]

                # stop qid - tid pair matching if cost not low enough
                if len(matched_qpts)<4:
                    continue

                # last check through direct calculation of sat and cam masks after Affine
                # transformation
                AfM_direct = lstsq_affine_matrix(matched_tpts, matched_qpts)
                t_warped = cv2.warpAffine(self.gt_mask, AfM_direct, frame.shape[:2][::-1])
                masks_iou = get_iou_cost(t_warped, q_mask)
                if masks_iou>=0.3:
                    loc_ok = True
                    break

            if loc_ok:
                break

        if loc_ok:
            AfM_inv = lstsq_affine_matrix(matched_qpts, matched_tpts)  # type: ignore
            if self.kalman:
                affine_matrix_inv1 = AfM_inv.copy()
                if not start:
                    filter.x = np.array([[affine_matrix_inv1[0][2], 0, affine_matrix_inv1[1][2], 0, affine_matrix_inv1[0][0], affine_matrix_inv1[0][1], affine_matrix_inv1[1][0], affine_matrix_inv1[1][1]]]).T
                    start = True

                z =  np.array([[affine_matrix_inv1[0][2] , affine_matrix_inv1[1][2] , affine_matrix_inv1[0][0], affine_matrix_inv1[0][1], affine_matrix_inv1[1][0], affine_matrix_inv1[1][1]]])

                filter.update(z)
                if frame_id > 5:
                    affine_matrix_inv1[0][2] = filter.x[0]
                    affine_matrix_inv1[1][2] = filter.x[2]
                    affine_matrix_inv1[0][0] = filter.x[4]
                    affine_matrix_inv1[0][1] = filter.x[5]
                    affine_matrix_inv1[1][0] = filter.x[6]
                    affine_matrix_inv1[1][1] = filter.x[7]

            else:
                affine_matrix_inv1 = AfM_inv.copy()

            sat_center, gps_data = get_gps_data(im_w, im_h, AfM_inv, affine_matrix_inv1)

            t_canv = np.zeros(self.map_image.shape, dtype=np.uint8)
            t_canv = cv2.warpAffine(frame, affine_matrix_inv1, t_canv.shape[:2][::-1])
            t_canv = np.where(t_canv==0, self.map_image, t_canv)

            cv2.rectangle(
                img=t_canv,
                pt1=(int(sat_center[0] - 4), int(sat_center[1] - 4)),
                pt2=(int(sat_center[0] + 4), int(sat_center[1] + 4)),
                color=(128, 0, 0),
                thickness=4
            )
        else:
            filter.predict()
            try:
                affine_matrix_inv1[0][2] = filter.x[0]  # type: ignore
                affine_matrix_inv1[1][2] = filter.x[2]  # type: ignore
                affine_matrix_inv1[0][0] = filter.x[4]  # type: ignore
                affine_matrix_inv1[0][1] = filter.x[5]  # type: ignore
                affine_matrix_inv1[1][0] = filter.x[6]  # type: ignore
                affine_matrix_inv1[1][1] = filter.x[7]  # type: ignore
                t_canv = np.zeros(self.map_image.shape, dtype=np.uint8)
                t_canv = cv2.warpAffine(raw_frame, affine_matrix_inv1, t_canv.shape[:2][::-1])  # type: ignore
                t_canv = np.where(t_canv==0, self.map_image, t_canv)
            except UnboundLocalError:
                pass
            else:
                t_canv = self.map_image.copy()

        return t_canv, gps_data

