/** Render the existing Pet ReID response contract as well as simple embedding-service adapters. */
export function commitOutcome(commit: Record<string, any> | null): string {
  if (!commit) return '';
  const gallery = commit.gallery ?? {};
  if (commit.action === 'enroll') {
    const added = gallery.added ?? gallery.added_image_ids?.length ?? commit.images?.length ?? 0;
    return `已为 ${commit.pet_id} 建档，录入 ${added} 张照片。`;
  }
  if (gallery.accepted === false) return '比对未通过，请检查照片或重新采集。';
  const top = gallery.candidates?.[0] ?? gallery.matches?.[0];
  const id = gallery.predicted_display_name || gallery.predicted_pet_id || top?.display_name || top?.pet_id;
  if (!id) return '比对已完成，没有找到足够相似的已建档狗狗。';
  const value = gallery.top1_score ?? top?.score;
  const score = typeof value === 'number' ? `（相似度 ${value.toFixed(2)}）` : '';
  return `最相近：${id}${score}`;
}
