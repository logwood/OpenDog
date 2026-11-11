import { expect, it } from 'vitest';
import { commitOutcome } from '../../h5/src/lib/outcome';

it('renders the real recognizer candidates contract without claiming an empty gallery', () => {
  expect(commitOutcome({ action: 'identify', gallery: { accepted: true, predicted_pet_id: 'dog-1',
    top1_score: 0.993, candidates: [{ pet_id: 'dog-1', score: 0.993 }] } })).toContain('最相近：dog-1');
  expect(commitOutcome({ action: 'identify', gallery: { accepted: false,
    candidates: [{ pet_id: 'dog-1', score: 0.4 }] } })).toContain('未通过');
});

it('reports newly inserted images rather than counting duplicate submissions as new enrollment', () => {
  expect(commitOutcome({ action: 'enroll', pet_id: 'dog-1', images: ['a', 'b'],
    gallery: { added_image_ids: [], duplicate_image_ids: ['a', 'b'] } })).toContain('录入 0 张');
});
