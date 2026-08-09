"""Add prediction status

Revision ID: 711b4a6463d1
Revises: 9f4ee9c4858b
Create Date: 2026-08-09 13:48:18.972743

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '711b4a6463d1'
down_revision = '9f4ee9c4858b'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('prediction', schema=None) as batch_op:
        batch_op.add_column(sa.Column('status', sa.String(length=20), nullable=True))

    op.execute(sa.text("""
        UPDATE prediction
        SET status = CASE
            WHEN disease = 'Background without leaves' THEN 'no_leaf'
            WHEN confidence < 60 THEN 'uncertain'
            WHEN LOWER(disease) LIKE '%healthy%' THEN 'healthy'
            ELSE 'diseased'
        END
    """))

    with op.batch_alter_table('prediction', schema=None) as batch_op:
        batch_op.alter_column(
            'status',
            existing_type=sa.String(length=20),
            nullable=False,
        )
        batch_op.create_index(batch_op.f('ix_prediction_status'), ['status'], unique=False)


def downgrade():
    with op.batch_alter_table('prediction', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_prediction_status'))
        batch_op.drop_column('status')
