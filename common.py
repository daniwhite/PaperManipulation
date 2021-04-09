def get_contact_point_from_results(contact_results, ll_idx, finger_idx):
        contact_point = None
        for i in range(contact_results.num_point_pair_contacts()):
            point_pair_contact_info = \
                contact_results.point_pair_contact_info(i)

            a_idx = int(point_pair_contact_info.bodyA_index())
            b_idx = int(point_pair_contact_info.bodyB_index())

            if ((a_idx == ll_idx) and (b_idx == finger_idx) or
                    (a_idx == finger_idx) and (b_idx == ll_idx)):
                contact_point = point_pair_contact_info.contact_point()

        return contact_point
